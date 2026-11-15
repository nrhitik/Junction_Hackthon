"""ClassicalPolicy: obs -> action wrapper around perception + planner (rung 1, sort_fixed).

    pol = ClassicalPolicy()                   # config.yaml, scene.yaml, calibration.yaml
    pol.reset({"instruction": obs["instruction"]})
    while True:
        action = pol.act(obs)                 # {"arm_q_target": (k,14), "hand_q_target": (k,14), "done": bool}
        obs, info = env.step(action)

Behaviour: go home -> settle -> look (head RGB-D) -> plan one object -> execute -> back home -> look again ...
until nothing sortable is left (then `done`). While it is planning (background thread when `threaded=True`) or
waiting for an operator confirm, act() returns 1-tick HOLD actions (the last target), so the env's watchdog
never fires. Targets are absolute, planned at <= motion.arm_max_vel and pre-checked against a SafetyFilter.

Operator confirm (real robot): ClassicalPolicy(require_confirm=True) asks before the first move to home and before
each object; the runner calls confirm() / skip() / stop() while `pol.waiting` is True (`pol.pending_text` says what).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.tasks import FIXED_INSTRUCTION, FIXED_MAPPING
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Planner

HZ = 30.0


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


_BIN_WORDS = {"left": "left", "center": "center", "centre": "center", "middle": "center", "right": "right"}
_COLOR_WORDS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "violet", "pink", "black", "white")


def parse_fixed_instruction(text):
    """Team: deterministic parse of the sort_fixed sentence pattern, e.g. "Put the yellow objects in the left bin, the
    purple objects in the center bin and the orange objects in the right bin." -> [(colour, None, bin), ...].
    Only when EVERY colour word in the text is matched to exactly one bin and nothing else (shapes, negation,
    "respectively") is said; otherwise None -> the LLM brain reads it."""
    import re
    t = text.lower()
    colors = "|".join(_COLOR_WORDS)
    pairs = re.findall(rf"\b({colors})\s+(?:objects?|ones?|items?|pieces?)\s+(?:go\s+)?(?:in|into|to|inside)\s+"
                       rf"the\s+(left|center|centre|middle|right)\s+(?:bin|box)", t)
    named = set(re.findall(rf"\b({colors})\b", t))
    if len(pairs) < 2 or {c for c, _ in pairs} != named or len({c for c, _ in pairs}) != len(pairs):
        return None
    if re.search(r"\b(cubes?|cylinders?|prisms?|triang\w*|except|not|respectively|unless|rest|other)\b", t):
        return None
    return [(c, None, _BIN_WORDS[b]) for c, b in pairs]


class _RulesPlan:
    """Minimal plan with the brain.instruction.Plan interface, for directly parsed rules."""

    def __init__(self, rules):
        self.rules, self.issues, self.question = list(rules), [], None

    def bin_for(self, color, shape=None):
        for key in ((color, shape), (None, shape), (color, None), (None, None)):
            hits = [b for c, s, b in self.rules if (c, s) == key]
            if hits:
                return hits[-1]
        return None


class ClassicalPolicy:
    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        self._bins0 = {k: list(map(float, v)) for k, v in self.scene.cfg["bins"]["centers"].items()}  # team: model bins
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        # team safety layer (self-collision + no cross-body reach): checked on every plan and every sent chunk
        self.team_safety = None
        if (self.cfg.get("team_safety") or {}).get("enabled", False):
            from .team_safety import TeamSafety
            self.team_safety = TeamSafety(self.cfg, ArmKinematics(self.cfg, self.spec))  # own pinocchio data
            self.planner.team_safety = TeamSafety(self.cfg, self.kin)
            for ts in (self.team_safety, self.planner.team_safety):
                ts.calibrate([(self.home, self.post["open"])])
        self.guard_trips = 0
        self.require_confirm = require_confirm
        self.threaded = threaded
        if threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        self._lid_cache = None
        try:
            if getattr(self.scene, "cfg", {}).get("lid"):
                self._lid_cache = self._lid_holes()
        except Exception as e:  # no lid in this scene file: task 3 leaves everything (0 points)
            self.log(f"[classical] lid holes unavailable: {e!r}")
            self._lid_cache = {}
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.mapping = info.get("mapping") or dict(FIXED_MAPPING)
        self.planner.mapping = self.mapping
        # team: task routing. sort_fixed = fixed colour map; sort_language = the brain (local LLM) reads the
        # instruction in a background thread; insert_shapes = each classified shape to its lid hole (scene.yaml)
        instr = info.get("instruction") or ""
        self.task = info.get("task") or ("insert_shapes" if "hole" in instr.lower() else
                                         "sort_fixed" if instr.strip() in ("", FIXED_INSTRUCTION) else "sort_language")
        self.brain = None
        # lid holes + their reach check are static: computed once in __init__ (a slow reset would trip the watchdog)
        self.planner.insert_holes = self._lid_cache if self.task == "insert_shapes" else None
        self._instruction = instr
        self.time_limit = info.get("time_limit", self.time_limit)
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.attempts = []          # (position, count) of objects we tried to pick
        self.failed = []            # positions whose plans failed
        self._handover = []         # team: (xy, colour, shape) of objects this policy put down for the other arm
        self._box_measured = False  # team: bin row position measured from head depth this episode
        self.pending = None         # Plan waiting for the operator
        self.pending_text = ""      # what the operator is asked to confirm
        self.waiting = False
        self.events = []
        self.plans = []
        self._job = None
        self._decision = None
        self.last_detections = []
        self.bonus_round = 0        # team final check: extra attempts granted after the table was re-checked
        self._lid_tuned = False
        self._lid_pose_done = False
        self._final_looked = False
        # team (organisers 2026-11-14): sort_fixed uses the REAL colours (yellow -> left, purple -> center,
        # orange -> right) and instructions may name colour, shape or both: ALWAYS read the instruction in the obs.
        # The fixed sentence pattern ("the X objects in the Y bin") is parsed directly; anything else -> the brain.
        instr = self._instruction
        if self.task in ("sort_fixed", "sort_language") and instr.strip():
            from .rules_parser import parse_rules   # team: rule-based reader (no LLM in the image)
            rules = parse_fixed_instruction(instr) or parse_rules(instr)
            if rules:
                self.brain = {"plan": _RulesPlan(rules), "done": True, "t0": time.monotonic(), "error": None}
                self.events.append((None, f"instruction parsed directly: {rules}"))
            elif (self.cfg.get("brain") or {}).get("enabled", True):
                self.brain = {"plan": None, "done": False, "t0": time.monotonic(), "error": None}
                threading.Thread(target=self._brain_job, args=(self.brain, instr), daemon=True).start()
            else:   # not understood and no LLM: sort nothing (0 points, never a wrong bin)
                self.brain = {"plan": None, "done": True, "t0": time.monotonic(), "error": "not understood"}
                self.events.append((None, f"instruction NOT understood (no LLM): {instr!r}: nothing is sorted"))

    def _apply_bin_shift(self, shift):
        """team: move the model's bins (and task-3 lid holes) `shift` m further from the robot (pelvis +x), from the
        ORIGINAL positions. Planner caches made with other bin positions are dropped."""
        sc = self.scene
        for k, v in self._bins0.items():
            sc.cfg["bins"]["centers"][k] = [v[0], v[1] + shift]     # mat y = pelvis x - 0.14
        if abs(shift - getattr(self, "_bin_shift", 0.0)) > 1e-4:
            self.planner.__dict__.pop("_spot_ok", None)
            self.planner.__dict__.pop("_raise_ok", None)
        self._bin_shift = shift
        self.planner.far_drop = self._far_drop_points(shift)
        if self.task == "insert_shapes" and self.planner.insert_holes:
            for hole in self.planner.insert_holes.values():
                hole["xy"] = np.asarray(hole["xy0"], float) + [shift, 0.0]

    def _far_drop_points(self, shift):
        """team: boxes measured far away (real setup: box centres ~60 cm) -> per bin the DEEPEST release point the
        arms reach: inset from the near inner wall (largest first; the object tips in once its centre of mass is
        past the wall's inner face) x height above the rim. {} when the boxes are at the model distance."""
        fd = self.cfg["grasp"].get("far_drop") or {}
        if not fd.get("enabled", True) or shift < float(fd.get("min_shift_m", 0.05)):
            return {}
        key = round(shift, 2)
        cache = self.__dict__.setdefault("_far_cache", {})
        if key in cache:
            return cache[key]
        pl, sc, kin = self.planner, self.scene, self.kin
        H, zc, wall = pl.height, sc.counter_z, float(sc.cfg["bins"]["wall_height"])
        g = self.cfg["grasp"]
        saved = g.get("drop_inset")
        out = {}
        try:
            for name in sc.cfg["bins"]["centers"]:
                c = pl.bin_xy(name)
                sides = ["left"] if c[1] > 0.1 else ["right"] if c[1] < -0.1 else ["left", "right"]
                best = None
                for inset in fd.get("insets", [0.035, 0.03, 0.025, 0.02, 0.015]):
                    g["drop_inset"] = inset
                    xy = pl.drop_xy(name)
                    for above in fd.get("heights", [0.06, 0.08, 0.10, 0.12]):
                        p = np.r_[xy, zc + wall + H / 2 + above]
                        for sd in sides:
                            q, ok, err, _ = kin.solve(sd, p, self.home, top_down=True,
                                                      extra_seeds=[self.home[ArmKinematics.sl(sd)]])
                            if ok:
                                best = {"xy": xy.copy(), "above": float(above), "inset": float(inset)}
                                break
                            if err <= float(fd.get("near_miss_m", 0.015)):
                                # near miss: take the point the hand DOES reach, if the object's centre is still
                                # past the inner wall by >= min_inset (it tips in); a few mm do not matter from 10 cm
                                r = kin.grasp_pose(sd, q)[0]
                                lo, hi = sc.bin_inner_box(name)
                                ends = [sc.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], sc.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]]
                                near_x = min(e[0] for e in ends)
                                if r[0] - near_x >= float(fd.get("min_inset_m", 0.01)):
                                    best = {"xy": r[:2].copy(), "above": float(r[2] - zc - wall - H / 2),
                                            "inset": float(r[0] - near_x)}
                                    break
                        if best:
                            break
                    if best:
                        break
                if best:
                    out[name] = best
                else:   # nothing reachable: aim at the reach limit is pointless -> no drop into this box
                    out[name] = {"xy": pl.drop_xy(name), "above": float(fd.get("heights", [0.12])[-1]),
                                 "inset": None}
        finally:
            g["drop_inset"] = saved
        cache[key] = out
        def txt(v):
            if v["inset"] is None:
                return "UNREACHABLE"
            return f"inset {100 * v['inset']:.1f} cm, {100 * v['above']:.0f} cm above rim"
        self.events.append((None, "far boxes: release points " + ", ".join(f"{k}: {txt(v)}" for k, v in out.items())))
        return out

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        if self.arm_cmd is None:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        if self.state == "init":
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._home_move = (a, h)
            dq = np.abs(self.home - self.arm_cmd).max()
            self.pending_text = f"move both arms to home_arm_q (max joint change {dq:.2f} rad, {len(a) / HZ:.1f} s)"
            if self.require_confirm:
                self.state, self.waiting, self.pending = "await_home", True, None
                self._note(t, f"waiting for confirm: {self.pending_text}")
                return self._hold()
            self.state = "home"
        if self.state == "await_home":
            d, self._decision = self._decision, None
            if d in ("skip", "stop"):
                self.waiting, self.state = False, "done"
                self._note(t, "operator stop before the first move")
                return self._hold(done=True)
            if d != "go":
                return self._hold()
            self.waiting, self.state = False, "home"
        if self.state == "home":
            a, h = self._home_move
            self._enqueue(a, h)
            self.state = "settle"
            self._note(t, f"to home ({len(a) / HZ:.1f} s)")
            return self._pop() if len(a) else self._hold()
        if self.state == "settle":
            n = int(round(self.cfg["motion"]["settle_s"] * HZ))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "look"
            return self._pop()
        if self.state == "look":
            lc = self.cfg["perception"].get("lid_camera") or {}
            if (self.task == "insert_shapes" and lc.get("enabled", False) and lc.get("look_pose") and not self._lid_tuned
                    and not self._lid_pose_done):
                # team: at the home pose the left hand hides the lid box from the head camera: move the hands out of
                # view (team-safety / clearance checked) before the one look that finds the lid
                self._lid_pose_done = True
                look = np.asarray(lc["look_pose"], float)
                a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, look, self.post["open"])
                why = self.planner._segment_ok("left", self.arm_cmd, self.hand_cmd, a, h) or                     self.planner._segment_ok("right", self.arm_cmd, self.hand_cmd, a, h)
                if why is None and len(a):
                    self._enqueue(a, h)
                    self.state = "executing"  # -> settle -> look (lid found from the look pose)
                    self._note(t, "task 3: hands out of view to find the lid")
                    return self._pop()
                self._note(t, f"task 3: lid look pose not safe ({why}), looking from here")
            if self.brain is not None and not self.brain["done"]:
                return self._hold()  # task 2: wait for the instruction plan (local LLM, ~0.4 s on the GPU)
            if self.threaded:
                if self._job is None:
                    self._job = {"obs": obs, "result": None}
                    threading.Thread(target=self._look_job, args=(self._job,), daemon=True).start()
                if self._job["result"] is None:
                    return self._hold()
                res, self._job = self._job["result"], None
            else:
                res = self._look(obs)
            return self._after_look(res, t)
        if self.state == "await":
            d, self._decision = self._decision, None
            if d == "go":
                self.waiting = False
                return self._start(self.pending, t)
            if d == "skip":
                self.waiting = False
                self.failed.append(self.pending.waypoints["grasp"][:2])
                self._note(t, f"operator skipped {self.pending.color}")
                self.state = "look"
                return self._hold()
            if d == "stop":
                self.waiting = False
                self.state = "done"
                self._note(t, "operator stop")
            return self._hold(done=self.state == "done")
        if self.state in ("exec_pick", "exec_place"):  # team feedback: segment done -> let the image settle, check
            n = max(1, int(round(float(self._fb().get("settle_s", 0.2)) * HZ)))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "verify_pick" if self.state == "exec_pick" else "verify_place"
            return self._pop()
        if self.state == "verify_pick":
            return self._verify_pick(obs, t)
        if self.state == "verify_place":
            return self._verify_place(obs, t)
        if self.state == "executing":  # queue drained -> back home: look again after settling
            self.state = "settle"
            return self.act(obs)
        return self._hold(done=True)

    # ------------------------------------------------------------------ team: head-camera feedback layer
    def _fb(self):
        return self.cfg["grasp"].get("feedback") or {}

    def _seen(self, obs, plan):
        """Target-colour 3D points from the head camera: (n near the hand's grasp point, n on the original spot,
        n inside the target bin). None if there is no image/depth."""
        rgb, depth = obs.get("head_rgb"), obs.get("head_depth")
        if rgb is None or depth is None:
            return None
        fb, sc = self._fb(), self.scene
        P = self.perception.color_points(rgb, depth, plan.color, step=int(fb.get("step", 2)))
        # team: the MEASURED arm (the real arm lags/sags 3-6 cm behind the command after the lift)
        q_meas = obs.get("arm_q")
        gp = self.kin.grasp_pose(plan.side, np.asarray(q_meas if q_meas is not None else self.arm_cmd, float))[0]
        n_hand = int((np.linalg.norm(P - gp, axis=1) < float(fb.get("hand_radius", 0.06))).sum())
        zc, H = sc.counter_z, self.planner.height
        g = plan.waypoints["grasp"][:2]
        on_spot = ((np.linalg.norm(P[:, :2] - g, axis=1) < float(fb.get("spot_radius", 0.035)))
                   & (P[:, 2] > zc + 0.005) & (P[:, 2] < zc + H + 0.01))
        n_bin = 0
        if plan.target_bin in sc.bin_names and len(P):
            lo, hi = sc.bin_inner_box(plan.target_bin)
            pm = np.asarray(sc.pelvis_to_mat(P))
            n_bin = int(np.all((pm >= lo - [0, 0, 0.01]) & (pm <= hi + [0, 0, 0.02]), axis=1).sum())
        return n_hand, int(on_spot.sum()), n_bin

    def _fingers_blocked(self, obs, plan):
        """team: finger contact = the commanded closed fingers are held open by an object (real robot: index
        commanded 1.07, stuck at 0.61 while holding the purple cube that the camera check called 'missed')."""
        hq = obs.get("hand_q")
        if hq is None or self.hand_cmd is None:
            return False
        sl = ArmKinematics.sl(plan.side)
        gap = np.abs(np.asarray(self.hand_cmd, float)[sl][3:] - np.asarray(hq, float)[sl][3:])
        return bool(gap.max() > float(self._fb().get("contact_gap_rad", 0.2)))

    def _verify_pick(self, obs, t):
        fb, cur = self._fb(), self._cur
        plan = cur["plan"]
        s = self._seen(obs, plan)
        mn = int(fb.get("min_px", 30))
        if s is None:
            verdict = "unknown"
        else:
            # sim: held -> 380-820 target px at the hand, not held -> 0. The original spot is mostly hidden under the
            # lifted hand, so it only serves as a log; `spot_required` restores the stricter old rule.
            n_hand, n_spot, _ = s
            if n_hand >= mn or self._fingers_blocked(obs, plan):
                verdict = "held"
            elif (fb.get("spot_required", False) or str(plan.shape or "").split("/")[0] in
                  (fb.get("spot_required_shapes") or [])) and n_spot < mn:
                verdict = "unknown"
            else:
                verdict = "missed"
        self._note(t, f"verify_pick {plan.color} {plan.shape or 'generic'}: {verdict} (hand/spot/bin px = {s})")
        if verdict == "missed":
            seg = self.planner.abort_segment(plan.side, self.arm_cmd, self.hand_cmd)
            if seg is not None:
                self._enqueue(*seg)
                self.state = "executing"  # -> home, settle, look: the object is re-detected and re-planned
                return self._pop()
            self._note(t, "abort segment not safe: finishing the plan")
        if verdict == "held" and self.task == "insert_shapes" and fb.get("inhand_correct", True):
            new = self._inhand_corrected(obs, plan, t)
            if new is not None:  # carry + release with the object's MEASURED centre over the hole
                ph = {n: (s_, e) for n, s_, e in new.phases}
                self._cur = {"plan": new, "i_pick": 0, "i_place": ph["retreat"][1], "shakes": 0}
                self._enqueue(new.arm[:ph["retreat"][1]], new.hand[:ph["retreat"][1]])
                self.state = "exec_place"
                return self._pop()
        self._enqueue(plan.arm[cur["i_pick"]:cur["i_place"]], plan.hand[cur["i_pick"]:cur["i_place"]])
        self.state = "exec_place"
        return self._pop()

    def _inhand_offset(self, obs, plan):
        """Team: where the held object actually is relative to the grasp point (pelvis xy, m), from head RGB-D.
        The hand hides the object's top: we see its sides. Across the view direction the centre is the middle of the
        visible extent; along it, the nearest visible face + half the object size. None if too few points."""
        fb = self._fb()
        rgb, depth = obs.get("head_rgb"), obs.get("head_depth")
        if rgb is None or depth is None:
            return None
        P = self.perception.color_points(rgb, depth, plan.color, step=1)
        gp = self.kin.grasp_pose(plan.side, self.arm_cmd)[0]
        P = P[np.linalg.norm(P - gp, axis=1) < float(fb.get("inhand_radius", 0.07))]
        if len(P) < int(fb.get("inhand_min_px", 60)):
            return None
        cam = self.perception.cal._T(self.perception.cam)[:3, 3]
        u = gp[:2] - cam[:2]
        u /= max(np.linalg.norm(u), 1e-9)
        v = np.array([-u[1], u[0]])
        s, w = P[:, :2] @ u, P[:, :2] @ v
        half = 0.5 * self.planner.height  # objects are as wide as they are high (scene.yaml)
        c_u = np.percentile(s, 5) + half
        c_v = 0.5 * (np.percentile(w, 5) + np.percentile(w, 95))
        centre = c_u * u + c_v * v
        return centre - gp[:2], gp

    def _inhand_corrected(self, obs, plan, t):
        fb = self._fb()
        r = self._inhand_offset(obs, plan)
        if r is None:
            self._note(t, "inhand: too few object points, release as planned")
            return None
        off, gp = r
        cap = float(fb.get("inhand_max_m", 0.03))
        n = float(np.linalg.norm(off))
        self._note(t, f"inhand offset est ({1000 * off[0]:+.1f},{1000 * off[1]:+.1f}) mm |{1000 * n:.1f}| "
                      f"gp ({gp[0]:.4f},{gp[1]:.4f})")
        if n < float(fb.get("inhand_min_m", 0.002)):
            return None
        if n > cap:
            off = off * (cap / n)
        drop = plan.waypoints.get("drop")
        if drop is None:
            return None
        hd = plan.waypoints.get("hole_dir")
        new = self.planner.place_plan(plan.side, self.arm_cmd, self.hand_cmd, np.asarray(drop[:2]) - off,
                                      float(drop[2]), plan.target_bin, plan.color, plan.shape,
                                      hint=None if hd is None else hd[:2], signed=(plan.shape or "").startswith("tri"))
        if not new.ok:
            self._note(t, f"inhand: corrected place not safe ({new.reason[:80]}), release as planned")
            return None
        new.waypoints["grasp"] = plan.waypoints["grasp"]
        self._note(t, f"inhand: release corrected by ({1000 * off[0]:+.1f},{1000 * off[1]:+.1f}) mm")
        return new

    def _verify_place(self, obs, t):
        fb, cur = self._fb(), self._cur
        plan = cur["plan"]
        s = self._seen(obs, plan)
        mn = int(fb.get("min_px", 30))
        stuck = s is not None and s[0] >= mn
        self._note(t, f"verify_place {plan.color} -> {plan.target_bin}: {'STUCK in hand' if stuck else 'released'} "
                      f"(hand/spot/bin px = {s})")
        if stuck and cur["shakes"] < int(fb.get("max_shakes", 2)):
            seg = self.planner.shake_segment(plan.side, self.arm_cmd, self.hand_cmd)
            if seg is not None:
                cur["shakes"] += 1
                self._enqueue(*seg)
                self.state = "exec_place"  # shake, settle, check again
                return self._pop()
            self._note(t, "shake segment not safe: going home")
        if plan.target_bin == "counter" and not stuck:
            # team hand-over: remember WHAT was put down where (the shape seen before the move is the trusted one;
            # a cube set down can read as a cylinder) and send this arm HOME before the other arm picks it
            # (the placing hand hovering over the spot blocked the other hand: self-collision)
            d = plan.waypoints.get("drop")
            if d is not None:
                self._handover.append([np.asarray(d[:2], float), plan.color, getattr(plan, "det_shape", None),
                                       getattr(plan, "handover_fin", None)])
        if self.cfg["policy"].get("look_from_retreat", False) and plan.target_bin != "counter":
            # team speed-up: the arm is above a bin, out of the table view -> look + plan the next pick from HERE
            # (no trip home + settle per pick). If nothing can be planned from here, _after_look goes home first.
            self._away_from_home = True
            self.state = "settle"
            return self._hold()
        self._enqueue(plan.arm[cur["i_place"]:], plan.hand[cur["i_place"]:])
        self.state = "executing"
        return self._pop()

    # ------------------------------------------------------------------ team: task 2 / task 3
    def _brain_job(self, box, instruction):
        try:
            from aaltonomous.brain.instruction import understand
            plan = understand(instruction, timeout=float((self.cfg.get("brain") or {}).get("timeout_s", 30.0)))
            box["plan"] = plan
            self.events.append((None, f"brain: {instruction!r} -> rules {plan.rules} issues {plan.issues} "
                                      f"({time.monotonic() - box['t0']:.1f} s)"))
        except Exception as e:  # LLM down: nothing is clear -> nothing is sorted (0 points, never a wrong bin)
            box["error"] = repr(e)
            self.events.append((None, f"brain error: {e!r}"))
        box["done"] = True

    SHAPE_NAME = {"triangle": "prism", "cube": "cube", "cylinder": "cylinder"}

    # colour words that may stand for each other when there is no exact rule (sim objects are red/blue, the real ones
    # and the new fixed instruction say orange/purple). Exact names always win.
    COLOR_STANDINS = {"red": ("orange",), "orange": ("red",), "blue": ("purple", "violet"),
                      "purple": ("blue", "violet"), "violet": ("purple", "blue")}

    def _bin(self, det):
        """Target bin for a detection under the current task, or None = leave it on the table."""
        if self.task == "insert_shapes":
            only = self.cfg["policy"].get("insert_only_shapes")  # team: e.g. [cube] -> other shapes stay (0, never -1)
            if only and det.shape not in only:
                return None
            return self.scene.lid_bin if det.shape in (self.planner.insert_holes or {}) else None
        if self.brain is not None:  # sort_fixed + sort_language: rules from the instruction in the obs
            plan = self.brain.get("plan")
            if not plan:
                return None
            shape = self.SHAPE_NAME.get(det.shape)
            named = {c for c, _, _ in getattr(plan, "rules", [])}
            if det.color not in named:
                # team: a stand-in name (red = orange, blue/violet = purple) only when the exact colour is not named
                for c in self.COLOR_STANDINS.get(det.color, ()):
                    if c in named:
                        return plan.bin_for(c, shape)
            return plan.bin_for(det.color, shape)
        return self.mapping.get(det.color)  # no instruction at all: the scene's fixed map

    def _lid_holes(self):
        """{shape: {xy, z, dirs}} in the pelvis frame from scene.yaml (sim). dirs = hole directions an object's
        pinch direction must match: square face normals (4), triangle flat -> apex (3); cylinder: any."""
        sc = self.scene
        z_top = sc.lid_z[1]
        out = {}
        for h in sc.lid_holes():
            cx, cy = h["center"]
            p0 = sc.mat_to_pelvis(np.array([cx, cy, z_top]))

            def pdir(ang):
                q = sc.mat_to_pelvis(np.array([cx + np.cos(ang), cy + np.sin(ang), z_top]))
                v = (q - p0)[:2]
                return v / np.linalg.norm(v)
            if h["shape"] == "cube":
                dirs = [pdir(k * np.pi / 2) for k in range(4)]
            elif h["shape"] == "triangle":  # apex = the outline vertex farthest from the centre (either lid layout)
                poly = np.asarray(h["poly"], float)
                v = poly[int(np.argmax(np.linalg.norm(poly - np.array([cx, cy]), axis=1)))] - np.array([cx, cy])
                dirs = [pdir(float(np.arctan2(v[1], v[0])) + k * 2 * np.pi / 3) for k in range(3)]
            else:
                dirs = None
            hole = {"xy": p0[:2], "xy0": p0[:2].copy(), "z": float(p0[2]), "dirs": dirs}
            # reach check once per episode: lowest release height (object bottom above the lid) the lid-bin arm can
            # get the grasp point to; unreachable holes are dropped -> those objects stay (0 points, never -1)
            side = "left" if p0[1] > 0 else "right"
            H = self.planner.height
            for dz in (self.cfg["grasp"].get("insert_drop_heights") or [0.02, 0.04, 0.06, 0.08, 0.10]):
                _, ok, _, _ = self.kin.solve(side, np.r_[p0[:2], p0[2] + H / 2 + dz], self.home, top_down=True,
                                             extra_seeds=[self.home[ArmKinematics.sl(side)]])
                if ok:
                    hole["dz"] = float(dz)
                    break
            if "dz" in hole:
                out[h["shape"]] = hole
            self.log(f"[classical] lid hole {h['shape']}: " +
                     (f"reachable, release {100 * hole['dz']:.0f} cm above the lid" if "dz" in hole else "OUT OF REACH (skipped)"))
        return out

    # ------------------------------------------------------------------ look + plan
    def _look_job(self, job):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        try:
            job["result"] = self._look(job["obs"])
        except Exception as e:  # never kill the control loop
            job["result"] = ("error", repr(e))

    def _look(self, obs):
        lc = self.cfg["perception"].get("lid_camera") or {}
        if self.task == "insert_shapes" and lc.get("enabled", False) and not self._lid_tuned and self._lid_cache:
            # team: tune the lid holes with the head camera (teammate's find_lid + head depth), once per episode,
            # each hole moved at most +-tolerance from the expected (scene) position
            self._lid_tuned = True
            try:
                from .perception import locate_lid_holes
                holes, rep = locate_lid_holes(obs.get("head_rgb"), obs.get("head_depth"), self.perception.cal,
                                              self.perception.cam, self._lid_cache,
                                              float(lc.get("tolerance_m", 0.029)))
                self.planner.insert_holes = holes
                for r in rep:
                    self.events.append((None, r))
            except Exception as e:
                self.events.append((None, f"lid camera tuning failed ({e!r}): expected holes kept"))
            if self._lid_pose_done and lc.get("look_pose"):
                return ("relook", None, 0)  # the lid look pose is not a planning pose: back home, look again
        bf = self.cfg["perception"].get("box_front") or {}
        if bf.get("enabled", False) and not self._box_measured:
            # team: measure where the bin row really stands (once per episode, from head depth) and move the bins
            # of the model there: bin exclusion, rim checks and drop points then match the real boxes
            self._box_measured = True
            try:
                pre = self.perception.detect(obs["head_rgb"], obs["head_depth"])   # objects first: not box points
                front, per = self.perception.measure_box_front(obs.get("head_depth"),
                                                               exclude_xy=[d.top[:2] for d in pre])
                sc = self.scene
                c0 = self._bins0["center"]
                model_front = sc.mat_to_pelvis(np.r_[c0[0], c0[1] - sc.cfg["bins"]["outer_size"][1] / 2, 0.0])[0]
                lo, hi = bf.get("shift_range_m", [-0.05, 0.15])
                shift = None if front is None else float(np.clip(front - model_front, lo, hi))
                if shift is not None and abs(shift) < float(bf.get("deadband_m", 0.015)):
                    shift = 0.0
                self._apply_bin_shift(shift or 0.0)
                self.events.append((None, f"box front measured: {per and {k: round(v, 3) for k, v in per.items()}} -> "
                                          f"{'model kept' if shift is None else f'bins moved {100 * shift:+.1f} cm'}"))
            except Exception as e:
                self.events.append((None, f"box front measurement failed ({e!r}): model bins kept"))
        dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        if (self.cfg["perception"].get("onnx") or {}).get("enabled", False):
            try:  # team: teammate's ONNX shape classifier fused with ours (perception.fuse_onnx_shapes)
                from .perception import fuse_onnx_shapes
                dets = fuse_onnx_shapes(dets, obs["head_rgb"], self.perception.size, self.cfg["perception"])
            except Exception as e:
                self.events.append((None, f"onnx shape fusion failed: {e!r}"))
        if self._handover:
            # team: an object at a hand-over spot keeps the shape it had before the move
            from .perception import _classify_shape
            for d in dets:
                for xy, col, shp, _ in self._handover:
                    if shp and d.color == col and np.linalg.norm(d.top[:2] - xy) < 0.05 and d.shape != shp \
                            and d.top_xy is not None:
                        self.events.append((None, f"hand-over: {col} object at ({xy[0]:+.2f},{xy[1]:+.2f}) seen as "
                                                  f"{d.shape}, kept as {shp}"))
                        _classify_shape(d, d.top_xy, self.perception.size, self.cfg["perception"].get("shape", {}),
                                        force=shp)
        self.last_detections = dets
        tol = float((self.cfg["grasp"].get("relocate") or {}).get("reuse_tol_m", 0.012))
        for rec in self._handover:
            # team speed: the other arm's plan for this spot was made when the spot was chosen (from home). Reuse it
            # if the object landed on the spot and both arms are home (that plan's start); once per hand-over.
            fin = rec[3]
            if fin is None or not getattr(fin, "ok", False) or fin.arm is None or not len(fin.arm):
                continue
            rec[3] = None
            near = [d for d in dets if d.color == rec[1] and np.linalg.norm(d.top[:2] - rec[0]) < tol]
            start_ok = (np.abs(np.asarray(fin.arm[0]) - self.arm_cmd).max() < 0.03
                        and np.abs(np.asarray(fin.hand[0]) - self.hand_cmd).max() < 0.15)
            if near and start_ok:
                fin.det_shape = rec[2]
                self.plans.append(fin)
                self.events.append((None, f"hand-over: object on the spot ({1000 * np.linalg.norm(near[0].top[:2] - rec[0]):.0f} mm"
                                          f" off), reusing the {fin.side} plan made with the spot"))
                return ("plan", fin, len(dets))
            self.events.append((None, f"hand-over: plan not reused ({'object off the spot' if not near else 'arms not at its start'})"
                                      ": planning again"))
        cands, tries = [], {}
        cap = self.cfg["policy"].get("max_attempts")  # team: 0 / null = no cap: retry until the episode time is up
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            n = sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < self.cfg["policy"]["same_object_radius"])
            if cap and n >= cap + self.bonus_round:
                continue
            tries[id(d)] = n
            cands.append(d)
        if self.cfg["policy"]["order"] == "shape_priority":
            # team: shapes in the configured order (default cube, cylinder, unknown, triangle), nearest bin first
            pri = self.cfg["policy"].get("shape_priority") or ["cube", "cylinder", None, "triangle"]

            def key(d):
                b = self._bin(d)
                shp = d.shape
                if shp is None and getattr(d, "shape_fit", None):
                    # team: not classified for sure, but its best fit says triangle -> treat it as a prism for the
                    # ORDER (prisms always last); the grasp still uses the generic grip
                    if max(d.shape_fit, key=d.shape_fit.get) == "triangle":
                        shp = "triangle"
                r = pri.index(shp) if shp in pri else pri.index(None) if None in pri else len(pri)
                return (r, np.inf if b is None else np.linalg.norm(d.top[:2] - self.planner.bin_xy(b)))
        elif self.cfg["policy"]["order"] == "nearest_bin":
            def key(d):
                b = self._bin(d)
                return np.inf if b is None else np.linalg.norm(d.top[:2] - self.planner.bin_xy(b))
        else:
            def key(d):
                return -d.n_px
        cands.sort(key=key)
        for d in cands:
            others = [o.top[:2] for o in dets if o is not d]
            b = self._bin(d)
            if b is None:  # not covered by the instruction / no matching hole: leave it (0 points, never -1)
                self.failed.append(d.top[:2])
                continue
            p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=b, obstacles=others,
                                  relocate=(
                                      self.time_limit - float(obs.get("t", 0.0))
                                      > float(self.cfg["policy"].get("relocate_min_time_s", 40))))
            self.plans.append(p)
            if p.ok:
                p.det_shape = d.shape  # team: the classified shape (hand-over memory)
                return ("plan", p, len(dets))
            self.failed.append(d.top[:2])
            self.log(f"[classical] skip {d}: {p.reason}")
        return ("none", None, len(dets))

    def _after_look(self, res, t):
        kind = res[0]
        if kind == "error":
            self._note(t, f"look/plan error {res[1]}: stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "relook":
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._enqueue(a, h)
            self.state = "executing"  # -> settle -> look (objects, from the home pose)
            self._note(t, "task 3: lid found, back home to look at the objects")
            return self._pop()
        if kind == "none" and getattr(self, "_away_from_home", False):
            # nothing plannable from above the bin: go home (the usual look pose) and look again before giving up
            self._away_from_home = False
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._enqueue(a, h)
            self.state = "executing"  # -> settle -> look
            self._note(t, "nothing plannable from the bin side: back home to look again")
            return self._pop()
        if kind == "none":
            # team final check: a fresh head-camera look before ending. Nothing outside the bins -> table clear.
            # Objects left -> list them; if time allows, ONE extra attempt each (once per episode), then check again.
            fc = self.cfg["policy"].get("final_check") or {}
            left = [f"{d.color} {d.shape or 'object'} at ({d.top[0]:+.2f},{d.top[1]:+.2f})" for d in self.last_detections]
            if not fc.get("enabled", True):
                self._note(t, f"nothing left to sort ({res[2]} detections)")
            elif not self._final_looked:
                self._final_looked = True
                self._note(t, f"final check: fresh look ({len(left)} objects seen outside the bins)")
                self.state = "settle"
                return self._hold()
            elif not left:
                self._note(t, "final check: table clear, nothing left outside the bins")
            elif self.bonus_round < int(fc.get("extra_attempts", 1)) and t + float(fc.get("min_time_s", 15)) < self.time_limit:
                self.bonus_round += 1
                self._final_looked = False
                self._note(t, f"final check: {len(left)} left ({'; '.join(left)}): one extra attempt each")
                self.state = "settle"
                return self._hold()
            else:
                self._note(t, f"final check: left on counter: {'; '.join(left)}")
            self.state = "done"
            return self._hold(done=True)
        plan = res[1]
        if t + plan.duration > self.time_limit:
            self._note(t, f"not enough time for {plan.color} ({plan.duration:.0f} s)")
            self.state = "done"
            return self._hold(done=True)
        if self.require_confirm:
            self.pending, self.waiting, self.state = plan, True, "await"
            self.pending_text = plan.summary()
            self._note(t, f"waiting for confirm: {plan.summary()}")
            return self._hold()
        return self._start(plan, t)

    def _start(self, plan, t):
        xy = plan.waypoints["grasp"][:2]
        self.attempts.append((xy, 1))
        ph = {name: (s, e) for name, s, e in plan.phases}
        self._note(t, f"execute {plan.summary()} grasp={plan.shape or 'generic'} finger_err={plan.finger_err_deg}")
        if self._fb().get("enabled") and "lift" in ph and "retreat" in ph:
            # team feedback layer: pick -> check (head camera) -> carry + release -> check -> home
            self._cur = {"plan": plan, "i_pick": ph["lift"][1], "i_place": ph["retreat"][1], "shakes": 0}
            self._enqueue(plan.arm[:ph["lift"][1]], plan.hand[:ph["lift"][1]])
            self.state = "exec_pick"
            return self._pop()
        self._enqueue(plan.arm, plan.hand)
        self.state = "executing"
        return self._pop()

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)

    def _pop(self):
        k = int(self.cfg["motion"]["chunk"])
        a, h = self.queue_a[:k], self.queue_h[:k]
        if self.team_safety is not None and len(a):
            # team guard: never send a chunk that the team safety layer rejects (plans are pre-checked; this catches
            # anything that slipped through, e.g. ticks between the pre-check samples)
            bad = self.team_safety.check_traj(a, h, every=3, start=(self.arm_cmd, self.hand_cmd))
            if bad:
                self.guard_trips += 1
                self.queue_a = self.queue_h = None
                self.events.append((None, f"GUARD stop: {bad[1]} (trip {self.guard_trips})"))
                self.log(f"[classical] GUARD stop: {bad[1]}")
                self.state = "done" if self.guard_trips >= 3 else "settle"
                return self._hold(done=self.state == "done")
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        if not len(a):
            return self._hold()
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _hold(self, done=False):
        """Hold the last target. While waiting (planning / operator) the hold is `motion.hold_ticks` ticks long, so a
        remote runner needs one request per hold chunk instead of one per tick: the command rate stays at 30 Hz
        even when the planner thread and other processes (sims) load the CPU."""
        k = 1 if done else max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
