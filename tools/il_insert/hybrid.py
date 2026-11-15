"""Rung-3 HYBRID: the classical pipeline picks and carries; a closed-loop controller does the last ~10 cm above the hole.

    pol = HybridInsertPolicy(controller=ExpertController(scene_client), recorder=Recorder(dir), ...)   # demos (sim)
    pol = HybridInsertPolicy(controller=LearnedController("bc.pt"), ...)                               # submission

Hand-over: after the team feedback layer's `verify_pick` says "held" (and, for task 3, the in-hand corrected place
plan was built), the place plan is CUT at the end of the horizontal carry: the hand is over the hole at carry height
(~10 cm above the lid). From there `final` runs every `step_s` (0.2 s): controller(features) -> (dx, dy, dz) for the
grasp point + release flag; each step is an IK straight-line segment (planner._cartesian, top-down, seeded from the
current pose -> keeps the wrist yaw the carry turned to the hole), time-scaled under the velocity caps and checked by
planner._segment_ok (clearance + safety-filter dry run). Release: hand to the release posture, retreat up, then the
team's verify_place (stuck -> shake) and home. Wrist yaw stays with the classical plan in this version.

Features (NOT privileged, the same at train and run time): grasp point - hole (xyz), the head-camera in-hand offset
estimate (team _inhand_offset, xy, + valid flag), step index, and a head RGB-D crop centred on the hole's projection.
"""
from __future__ import annotations

import dataclasses
import os
import time

import numpy as np

from aaltonomous.baselines.classical.planner import Plan, time_scale, ramp
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.policy import ClassicalPolicy, HZ

CROP_PX, CROP_OUT = 160, 64     # head-image crop around the hole's projection (px), resized for the network


@dataclasses.dataclass
class FinalCfg:
    step_s: float = 0.2          # one controller decision per 6 ticks
    max_step_m: float = 0.012    # per-decision grasp-point move cap
    max_steps: int = 45          # 9 s, then release wherever we are
    min_clear_m: float = 0.004   # never command the object bottom below lid top + this
    release_hold_s: float = 0.25
    retreat_up_m: float = 0.08


class HybridInsertPolicy(ClassicalPolicy):
    def __init__(self, *a, controller=None, recorder=None, final_cfg: FinalCfg | None = None, **kw):
        super().__init__(*a, **kw)
        self.controller, self.recorder = controller, recorder
        self.fcfg = final_cfg or FinalCfg()
        self.final_log = []          # one dict per final stage (for evaluation)

    # -------------------------------------------------------------- hand-over
    def _verify_pick(self, obs, t):
        out = super()._verify_pick(obs, t)
        if self.controller is None or self.task != "insert_shapes" or self.state != "exec_place":
            return out
        plan, cur = self._cur["plan"], self._cur
        if getattr(plan, "target_bin", None) == "counter":
            return out                             # team: hand-over to the other arm (relocation): no final stage
        hole = (self.planner.insert_holes or {}).get(str(plan.shape or "").split("/")[0]) if plan.shape else None
        if hole is None:
            hole = self._hole_for(plan)
        drop = plan.waypoints.get("drop")
        if hole is None or drop is None or self.queue_a is None:
            return out
        ph = {n: (s, e) for n, s, e in plan.phases}
        if "carry" not in ph:
            return out
        end = cur["i_place"]                       # the queue holds plan ticks [end - len(queue), end)
        q0 = end - len(self.queue_a) - (len(out["arm_q_target"]) if out and "arm_q_target" in out else 0)
        s, e = ph["carry"]
        cut = None
        for k in range(s, e):                      # first carry tick with the grasp point over the drop xy
            gp = self.kin.grasp_pose(plan.side, plan.arm[k])[0]
            if np.linalg.norm(gp[:2] - np.asarray(drop[:2])) < 0.01:
                cut = k + 1
                break
        if cut is None:
            return out
        keep = cut - (end - len(self.queue_a))
        if keep <= 0:
            return out
        self.queue_a, self.queue_h = self.queue_a[:keep], self.queue_h[:keep]
        self._final = {"plan": plan, "hole": hole, "step": 0, "t0": t, "drop": np.asarray(drop, float), "rec": [],
                       "inhand": None}
        self.state = "exec_carry"
        self._note(t, f"hybrid: carry cut at tick {cut} (of {e}), final stage next over the {plan.shape} hole")
        return out

    def _hole_for(self, plan):
        holes = self.planner.insert_holes or {}
        if not holes:
            return None
        d = plan.waypoints.get("drop")
        return min(holes.values(), key=lambda h: np.linalg.norm(np.asarray(h["xy"]) - np.asarray(d[:2]))) if d is not None else None

    def act(self, obs):
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        t = float(obs.get("t", 0.0))
        if self.state == "exec_carry":
            self.state = "final"
            r = self._inhand_offset(obs, self._final["plan"])   # head-camera estimate, once, at the hand-over
            self._final["inhand"] = None if r is None else np.asarray(r[0], float)
        if self.state == "final":
            return self._final_step(obs, t)
        if self.state == "exec_release":
            self.state = "exec_place"                           # -> settle -> team verify_place (stuck -> shake)
            return super().act(obs)
        return super().act(obs)

    # -------------------------------------------------------------- final stage
    def features(self, obs):
        f, plan = self._final, self._final["plan"]
        gp = self.kin.grasp_pose(plan.side, self.arm_cmd)[0]
        hole = np.r_[np.asarray(f["hole"]["xy"], float), float(f["hole"]["z"])]
        ih = f["inhand"]
        low = np.r_[gp - hole, (ih if ih is not None else np.zeros(2)), float(ih is not None),
                    f["step"] / self.fcfg.max_steps].astype(np.float32)
        return {"low": low, "img": self._crop(obs, hole), "gp": gp, "hole": hole}

    def _crop(self, obs, hole):
        import cv2
        rgb, depth = obs.get("head_rgb"), obs.get("head_depth")
        out = np.zeros((CROP_OUT, CROP_OUT, 4), np.uint8)
        if rgb is None:
            return out
        u, v, z = self.perception.cal.project(self.perception.cam, hole)
        if z <= 0:
            return out
        H, W = rgb.shape[:2]
        h = CROP_PX // 2
        x0, y0 = int(round(u)) - h, int(round(v)) - h
        pad = lambda a: np.pad(a, ((h, h), (h, h)) + ((0, 0),) * (a.ndim - 2))
        c = pad(np.asarray(rgb))[y0 + h:y0 + h + CROP_PX, x0 + h:x0 + h + CROP_PX]
        out[..., :3] = cv2.resize(np.ascontiguousarray(c), (CROP_OUT, CROP_OUT), interpolation=cv2.INTER_AREA)
        if depth is not None:   # depth relative to the hole's camera distance, +-12.7 cm -> 0..255
            d = pad(np.asarray(depth, float))[y0 + h:y0 + h + CROP_PX, x0 + h:x0 + h + CROP_PX]
            d = cv2.resize(d, (CROP_OUT, CROP_OUT), interpolation=cv2.INTER_NEAREST)
            rel = np.where(d > 0, (d - 1000 * z) / 1.0 + 127, 0)
            out[..., 3] = np.clip(rel, 0, 255).astype(np.uint8)
        return out

    def _final_step(self, obs, t):
        f, fc = self._final, self.fcfg
        plan = f["plan"]
        feat = self.features(obs)
        act = self.controller(feat, self, obs)          # {"d": (3,), "release": bool, "label": optional expert label}
        if self.recorder is not None:
            self.recorder.add(feat, act)
        f["step"] += 1
        release = bool(act["release"]) or f["step"] >= fc.max_steps
        if not release:
            d = np.clip(np.asarray(act["d"], float), -fc.max_step_m, fc.max_step_m)
            gp = feat["gp"]
            tgt = gp + d
            H = self.planner.height
            tgt[2] = max(tgt[2], f["hole"]["z"] + H / 2 + fc.min_clear_m)   # object bottom stays above the lid
            seg = self._servo(plan.side, gp, tgt)
            if seg is None:   # unsafe / no IK: release where we are
                self._note(t, f"hybrid: step {f['step']} not safe -> release here")
                release = True
            else:
                self._enqueue(*seg)
                return self._pop()
        return self._release(obs, t, feat)

    def _servo(self, side, p0, p1):
        pl, mo = self.planner, self.cfg["motion"]
        dummy = Plan(ok=True)
        qs = pl._cartesian(side, p0, p1, self.arm_cmd, "final", dummy) if np.linalg.norm(p1 - p0) > 1e-4 else []
        if qs is None:
            return None
        n = max(1, int(round(self.fcfg.step_s * HZ)))
        a = time_scale([self.arm_cmd] + list(qs), mo["arm_max_vel"], mo["arm_max_acc"], 1.0 / HZ) if qs else \
            np.repeat(self.arm_cmd[None], 0, 0)
        if len(a) < n:
            a = np.vstack([a, np.repeat((a[-1] if len(a) else self.arm_cmd)[None], n - len(a), 0)])
        h = np.repeat(self.hand_cmd[None], len(a), 0)
        why = pl._segment_ok(side, self.arm_cmd, self.hand_cmd, a, h)
        return None if why else (a, h)

    def _release(self, obs, t, feat):
        f, fc, pl, mo = self._final, self.fcfg, self.planner, self.cfg["motion"]
        plan = f["plan"]
        sl = ArmKinematics.sl(plan.side)
        g = self.cfg["grasp"]
        h_rel = self.hand_cmd.copy()
        h_rel[sl] = self.post[g.get("release_posture", "release")][sl]
        h1 = ramp(self.hand_cmd, h_rel, mo["hand_max_vel"], 1.0 / HZ)
        h1 = np.vstack([h1, np.repeat(h_rel[None], int(round(fc.release_hold_s * HZ)), 0)]) if len(h1) else h_rel[None]
        a1 = np.repeat(self.arm_cmd[None], len(h1), 0)
        gp = feat["gp"]
        qs = pl._cartesian(plan.side, gp, gp + [0, 0, fc.retreat_up_m], self.arm_cmd, "final_retreat", Plan(ok=True)) or []
        a2 = time_scale([self.arm_cmd] + list(qs), mo["arm_max_vel"], mo["arm_max_acc"], 1.0 / HZ) if qs else a1[:0]
        h2 = np.repeat(h_rel[None], len(a2), 0)
        # home from wherever we end: the team's verify_place enqueues plan.arm[i_place:] -> give it OUR home path
        q_end = a2[-1] if len(a2) else self.arm_cmd
        home = q_end.copy(); home[sl] = self.home[sl]
        h_open = h_rel.copy(); h_open[sl] = self.post["open"][sl]
        a3, h3 = pl.go_to(q_end, h_rel, home, h_open)
        self._cur = {"plan": dataclasses.replace(plan, arm=np.asarray(a3), hand=np.asarray(h3)), "i_pick": 0, "i_place": 0,
                     "shakes": 0}
        hz = feat["gp"] - feat["hole"]
        self.final_log.append({"t": round(t, 2), "shape": plan.shape, "side": plan.side, "steps": f["step"],
                               "target": getattr(self.controller, "target", None), "color": plan.color,
                               "release_rel_mm": np.round(1000 * hz, 1).tolist(), "inhand": None if f["inhand"] is None
                               else np.round(1000 * f["inhand"], 1).tolist()})
        if self.recorder is not None:
            self.recorder.end_stage(self.final_log[-1])
        self._note(t, f"hybrid: release after {f['step']} steps, grasp point - hole = {np.round(1000 * hz, 1)} mm")
        self._enqueue(np.vstack([a1, a2]) if len(a2) else a1, np.vstack([h1, h2]) if len(a2) else h1)
        self.state = "exec_release"
        return self._pop()


# ------------------------------------------------------------------ controllers
class ExpertController:
    """PRIVILEGED (sim only, demos): servo the TRUE object centre over the hole, descend, release just above the lid.
    `noise_m` perturbs the EXECUTED move (DART-style), the label is always the clean expert action."""

    def __init__(self, scene_client, noise_m=0.0, tol_xy=0.0025, release_clear=0.010, rng=None):
        self.sc, self.noise, self.tol, self.rel_clear = scene_client, noise_m, tol_xy, release_clear
        self.rng = rng or np.random.default_rng()
        self.target = None

    def _obj(self, pol, gp):
        objs = [o for o in self.sc.state()["objects"] if o["active"]]
        plan = pol._final["plan"]
        if pol._final["step"] == 0 or self.target is None:
            c = [o for o in objs if o["color"] == plan.color] or objs
            self.target = min(c, key=lambda o: np.linalg.norm(np.asarray(o["pos"]) - gp))["name"]
        return next((o for o in objs if o["name"] == self.target), None)

    def __call__(self, feat, pol, obs):
        gp, hole = feat["gp"], feat["hole"]
        o = self._obj(pol, gp)
        if o is None:
            return {"d": np.zeros(3), "release": True, "label": np.r_[np.zeros(3), 1.0]}
        p = np.asarray(o["pos"], float)
        H = pol.planner.height
        e = hole[:2] - p[:2]                         # move the OBJECT centre onto the hole
        clear = (p[2] - H / 2) - hole[2]             # object bottom above the lid top
        ne = float(np.linalg.norm(e))
        dz = 0.0
        if ne < 0.006:
            dz = -min(max(clear - 0.008, 0.0), 0.012)
        elif ne < 0.015:
            dz = -min(max(clear - 0.03, 0.0), 0.006)
        release = ne < self.tol and clear < self.rel_clear + 0.004
        d = np.r_[np.clip(e, -0.012, 0.012), dz]
        label = np.r_[d, float(release)]
        if self.noise > 0 and not release:
            d = d + np.r_[self.rng.normal(0, self.noise, 2), self.rng.normal(0, 0.3 * self.noise)]
        return {"d": d, "release": release, "label": label, "true_err_mm": np.round(1000 * np.r_[e, clear], 1)}


class Recorder:
    """Saves one .npz per final stage: features (low, img), expert labels, outcome filled in by the runner."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.buf, self.stages = [], []

    def add(self, feat, act):
        self.buf.append((feat["low"], feat["img"], np.asarray(act.get("label", np.r_[act["d"], float(act["release"])]),
                                                               np.float32)))

    def end_stage(self, info):
        if not self.buf:
            return
        name = os.path.join(self.dir, f"stage_{int(time.time() * 1000)}.npz")
        np.savez_compressed(name, low=np.stack([b[0] for b in self.buf]), img=np.stack([b[1] for b in self.buf]),
                            label=np.stack([b[2] for b in self.buf]))
        self.stages.append({"file": name, **info})
        self.buf = []
