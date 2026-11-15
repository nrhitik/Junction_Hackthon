"""Pick-and-place planning for one detected object: Cartesian waypoints -> IK -> 30 Hz joint trajectories.

Sequence (one arm; the other holds its current target):
  home -> pregrasp at carry height (joint space, hand to preshape) -> descend (Cartesian) -> close -> lift -> carry
  above the bin (Cartesian, else joint space direct / via a raised midpoint) -> lower to drop height -> release
  (hand.release) -> retreat up -> home (hand back to open)
Each side is tried with every grasp.finger_hints direction (fingers forward can touch the centre bin's wall).
Cross-body (neither arm can do object -> bin): a RELOCATION plan picks with one arm and puts the object down on
the counter at a spot from which the bin-side arm has a full plan; the policy's next look picks it up from there.
Every planned tick is checked before it is used:
  * per-joint speed <= motion.arm_max_vel (trapezoidal time scaling on the inf-norm path length),
  * hand points (palm, grasp point, finger links) over the counter footprint stay above the counter
    (counter_clearance in transit; fingertip_clearance during the grasp itself),
  * hand points over a bin footprint (+1 cm) stay above its rim, except while releasing,
  * a dry run through a fresh SafetyFilter (same config as the env) produces 0 clipped ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter
from aaltonomous.interface.tasks import FIXED_MAPPING

from .kinematics import ArmKinematics


@dataclass
class Plan:
    ok: bool
    side: str | None = None
    color: str = ""
    reason: str = ""
    target_bin: str | None = None
    waypoints: dict = field(default_factory=dict)      # name -> grasp point (pelvis)
    ik: dict = field(default_factory=dict)             # name -> (ok, pos_err, tilt_deg)
    arm: np.ndarray | None = None                      # (T, 14)
    hand: np.ndarray | None = None                     # (T, 14)
    phases: list = field(default_factory=list)         # (name, start tick, end tick)
    safety_clips: int | None = None
    shape: str | None = None                           # team: grasp mode used ("cube" | "triangle" | None = generic)
    finger_err_deg: float | None = None                # team: achieved finger direction vs the shape's pinch axis

    @property
    def duration(self):
        return 0.0 if self.arm is None else len(self.arm) / 30.0

    def summary(self):
        if not self.ok:
            return f"{self.color:<6} SKIP ({self.reason})"
        wp = " ".join(f"{k}=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for k, v in self.waypoints.items()
                      if k in ("grasp", "drop"))
        tilt = max(t for _, _, t in self.ik.values())
        err = max(e for _, e, _ in self.ik.values()) * 1000
        return (f"{self.color:<6} {self.side:<5} arm -> {self.target_bin} bin  {wp}  IK ok max_err={err:.1f}mm "
                f"max_tilt={tilt:.0f}deg  {self.duration:.1f}s  safety_clips={self.safety_clips}")


def time_scale(Qs, vmax, amax, dt):
    """Piecewise-linear joint path through Qs (n, d) -> samples (T, d) at dt, EXCLUDING Qs[0], with a trapezoidal
    profile on the inf-norm arc length: no joint ever moves faster than vmax."""
    Qs = np.asarray(Qs, float)
    seg = np.abs(np.diff(Qs, axis=0)).max(axis=1)
    S = float(seg.sum())
    if S < 1e-9:
        return np.empty((0, Qs.shape[1]))
    ta = vmax / amax
    if amax * ta * ta > S:                       # triangle profile
        ta = np.sqrt(S / amax)
        vpk = amax * ta
        T = 2 * ta
    else:
        vpk = vmax
        T = 2 * ta + (S - amax * ta * ta) / vmax
    n = max(1, int(np.ceil(T / dt)))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * amax * t ** 2,
                 np.where(t <= T - ta, 0.5 * amax * ta ** 2 + vpk * (t - ta), S - 0.5 * amax * (T - t) ** 2))
    s = np.clip(s, 0, S)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((n, Qs.shape[1]))
    for k, sk in enumerate(s):
        i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-12 else (sk - cum[i]) / seg[i]
        out[k] = Qs[i] + f * (Qs[i + 1] - Qs[i])
    return out


def ramp(a, b, vmax, dt):
    """Linear move a -> b (excluding a) with every component at most vmax."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.ceil(np.abs(b - a).max() / (vmax * dt) + 1e-9))
    if n == 0:
        return np.empty((0, len(a)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


class Planner:
    def __init__(self, cfg, scene, postures, home, kin: ArmKinematics | None = None, safety: SafetyConfig | None = None,
                 spec: JointSpec | None = None, mapping=None, dt=1.0 / 30.0):
        self.cfg, self.scene, self.post, self.home = cfg, scene, postures, np.asarray(home, float)
        self.spec = spec or JointSpec()
        self.kin = kin or ArmKinematics(cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=dt)
        self.dt = dt
        self.height = float(cfg["perception"].get("object_height") or scene.object_height())
        self.mapping = mapping or dict(FIXED_MAPPING)  # sort_fixed: fixed colour -> bin position

    # ------------------------------------------------------------------ geometry helpers
    def bin_xy(self, name):
        return self.scene.mat_to_pelvis(np.r_[self.scene.cfg["bins"]["centers"][name], 0.0])[:2]

    def drop_xy(self, name):
        """Release point (pelvis xy) in bin `name`: the point of the inner footprint nearest the pelvis, `grasp.drop_inset`
        inside the inner walls (the published box row puts the box centres past the arm's reach). None -> centre."""
        inset = self.cfg["grasp"].get("drop_inset")
        c = self.bin_xy(name)
        if inset is None:
            return c
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        lo2 = np.minimum(np.minimum(a, b) + inset, c)
        hi2 = np.maximum(np.maximum(a, b) - inset, c)
        return np.clip(np.zeros(2), lo2, hi2)

    def _over_counter(self, p):
        pm = self.scene.pelvis_to_mat(p)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        return np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)

    def bin_clearance(self, side, arm_q, hand_q, margin=None):
        """Lowest height (m) above the bin rims of the hand points over a bin's outer footprint (+margin), inf if none.
        Bins are static boxes in the sim and on the table: only the drop itself may go below a rim."""
        if margin is None:  # team: footprint margin around the bins (config grasp.bin_margin, was a fixed 1 cm)
            margin = float(self.cfg["grasp"].get("bin_margin", 0.01))
        pts = self.kin.hand_points(side, arm_q, hand_q)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + margin
        rim = self.scene.counter_z + float(b["wall_height"])
        out = np.inf
        lid_in = None
        if getattr(self, "insert_holes", None) and self.scene.lid_height == "floor":
            # team: lid on the bin floor -> the hand must reach INTO the lid bin. Points well inside its inner walls
            # (>= 1.5 cm) only have to stay above the lid; points near the walls still have to clear the rim.
            inner = np.asarray(b["outer_size"], float) / 2 - float(b["wall_thickness"]) - 0.015
            lid_in = (self.scene.lid_bin, inner, self.scene.counter_z + self.scene.lid_z[1])
        for name, c in b["centers"].items():
            d = np.abs(pm[:, :2] - np.asarray(c, float))
            over = np.all(d <= half, axis=1)
            if not over.any():
                continue
            if lid_in is not None and name == lid_in[0]:
                inside = over & np.all(d <= lid_in[1], axis=1)
                if inside.any():
                    out = min(out, float((pts[inside, 2] - lid_in[2]).min()))
                over = over & ~inside
                if not over.any():
                    continue
            out = min(out, float((pts[over, 2] - rim).min()))
        return out

    def clearance(self, side, arm_q, hand_q, fingers_only=False):
        """Lowest height (m) above the counter surface of the hand points that are over the counter (inf if none)."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        if fingers_only:
            pts = pts[2:]
        over = self._over_counter(pts)
        if not over.any():
            return np.inf
        return float((pts[over, 2] - self.scene.counter_z).min())

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, side, p, seed, name, plan):
        q, ok, err, tilt = self.kin.solve(side, p, seed, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        plan.ik[name] = (ok, err, float(np.degrees(tilt)))
        plan.waypoints[name] = np.asarray(p, float)
        return q, ok

    def _cartesian(self, side, p0, p1, q0, name, plan):
        """IK waypoints along the straight line p0 -> p1 (excluding p0), seeded continuously. A step whose joints
        jump more than `max_joint_step` is bisected (up to 3 times) before it counts as a failure."""
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / self.cfg["motion"]["cart_step"])))
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qs, q, info = [], q0, [True, 0.0, 0.0]
        jmax = float(self.cfg["motion"].get("max_joint_step", 0.4))

        def seg(pa, pb, q, depth):
            qn, ok, err, tilt = self.kin.solve(side, pb, q, top_down=True, restarts=False)
            if ok and np.abs(qn - q).max() <= jmax:
                info[1], info[2] = max(info[1], err), max(info[2], float(np.degrees(tilt)))
                return [qn]
            if depth >= 3:
                plan.reason = plan.reason or (f"IK jump on {name}" if ok else "")
                return None
            pm = 0.5 * (pa + pb)
            a = seg(pa, pm, q, depth + 1)
            b = a and seg(pm, pb, a[-1], depth + 1)
            return a + b if b else None

        for k in range(1, n + 1):
            out = seg(p0 + (p1 - p0) * (k - 1) / n, p0 + (p1 - p0) * k / n, q, 0)
            if out is None:
                plan.ik[name] = (False, info[1], info[2])
                return None
            qs += out
            q = out[-1]
        plan.ik[name] = (True, info[1], info[2])
        plan.waypoints[name] = p1
        return qs

    def _joint_carry(self, side, q_from, p_from, p_to, z_min, name, plan):
        """Joint-space move to an IK solution at p_to (direct, else via a raised midpoint), accepted if the grasp
        point never dips below z_min on the way."""
        def low(qa, qb):
            return any(self.kin.grasp_pose(side, qa + f * (qb - qa))[0][2] < z_min for f in np.linspace(0, 1, 15))

        q_to, ok = self._ik(side, p_to, q_from, name, plan)
        if ok and not low(q_from, q_to):
            return [q_to]
        for dz in (0.06, 0.12):
            mid = 0.5 * (np.asarray(p_from, float) + p_to) + [0, 0, dz]
            q_mid, ok1 = self._ik(side, mid, q_from, name + "_mid", plan)
            if not ok1 or low(q_from, q_mid):
                continue
            q_to, ok2 = self._ik(side, p_to, q_mid, name, plan)
            if ok2 and not low(q_mid, q_to):
                plan.reason = ""
                return [q_mid, q_to]
        plan.ik[name] = (False, 0.0, 0.0)
        plan.reason = f"{name}: no joint-space path above the bin rim"
        return None

    # ------------------------------------------------------------------ main
    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        """team wrapper. Task 3: the lid bin is reached by ONE arm only (left bin -> left arm). An object on the other
        half of the table goes straight to the hand-over: the near arm puts it down at a spot (grasp.relocate.
        insert_spots first) from which the lid arm inserts it. No slow direct attempts that cannot work."""
        hole = (getattr(self, "insert_holes", None) or {}).get(getattr(det, "shape", None))
        if hole is not None and bin_name is not None:
            by = self.bin_xy(bin_name)[1]
            relocate = relocate and self.cfg["grasp"].get("relocate", {}).get("enabled", True)
            far = float(self.cfg["grasp"].get("relocate", {}).get("insert_far_y", 0.05))
            if relocate and abs(by) > 0.1 and np.sign(det.top[1]) != np.sign(by) and abs(det.top[1]) > far:
                r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles)
                if not r.ok:   # direct plans cannot work from here (cross-body / lid bin out of reach): do not try
                    r.reason = f"hand-over failed: {r.reason}"
                return r
            if abs(by) > 0.1:
                # team task 3: only the lid-bin arm can insert (the other arm misses every hole by 10-14 cm):
                # plan that arm only; the other arm is used for the hand-over alone
                lid_side = "left" if by > 0 else "right"
                p = self._plan_side_hints(det, lid_side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float))
                if p.ok or not relocate:
                    return p
                r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles)
                if r.ok:
                    return r
                p.reason = f"{lid_side}: {p.reason}; hand-over: {r.reason}"
                return p
        return self._plan_once(det, arm_q, hand_q, bin_name, obstacles, relocate)

    def _plan_once(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        """Plan for Detection `det` starting from the commanded arm_q/hand_q (14,). Tries the preferred arm (the
        target bin's side; for the centre bin the object's side), then the other one. If neither arm can do the
        whole pick -> bin (cross-body: the arm that reaches the object cannot reach the bin), plans a RELOCATION
        instead (`grasp.relocate`): pick + place on the counter at a spot from which an arm can finish the job;
        the next look re-detects the object there. obstacles: (n, 2) pelvis xy of the other detected objects.
        Returns the first Plan that passes every check, else the last failure."""
        g = self.cfg["grasp"]
        bin_name = bin_name or self.mapping.get(det.color)
        by = self.bin_xy(bin_name)[1] if bin_name is not None else 0.0
        ref = by if abs(by) > 0.1 else det.top[1]
        side0 = "left" if ref > g["y_split"] else "right"
        if g.get("arm_select", "nearest") == "nearest":
            # team: the hand nearest the object (head-camera position vs each hand's current grasp point) goes first;
            # the other hand only when the near one has no valid plan (reach / bin / safety), then relocation
            d = {s: float(np.linalg.norm(self.kin.grasp_pose(s, np.asarray(arm_q, float))[0][:2] - det.top[:2]))
                 for s in ("left", "right")}
            side0 = min(d, key=d.get)
        last = Plan(False, color=det.color, reason=f"no bin for colour {det.color!r}")
        if bin_name is None:
            return last
        if g.get("hand_select", {}).get("enabled") and getattr(det, "shape", None) == "triangle":
            # team: prism -> try both hands x 3 pinch directions, take the best accuracy score
            p, _ = self.best_pinch_plan(det, np.asarray(arm_q, float), np.asarray(hand_q, float), bin_name)
            if p is not None:
                return p
        reasons, good = [], []
        for side in (side0, "right" if side0 == "left" else "left"):
            p = self._plan_side_hints(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float))
            if p.ok:
                if g.get("arm", "best") != "best":
                    return p
                good.append(p)
            else:
                extra = ""
                if os.environ.get("AALTO_PLAN_DEBUG"):  # team diag: which waypoints failed + where
                    extra = " {" + ", ".join(f"{k}:{'ok' if v[0] else 'X'}" for k, v in p.ik.items()) + "} wp=" +                         ", ".join(f"{k}=({v[0]:.2f},{v[1]:+.2f},{v[2]:.2f})" for k, v in p.waypoints.items() if np.size(v) == 3)
                reasons.append(f"{side}: {p.reason}{extra}")
                last = p
        if good:
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons)
        if relocate and g.get("relocate", {}).get("enabled", True):
            r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles)
            if r.ok:
                return r
            last.reason += f"; relocate: {r.reason}"
        return last

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Candidate counter spots (pelvis xy) for a relocation, nearest the target bin first."""
        rc = self.cfg["grasp"].get("relocate", {})
        sc = self.scene
        L = sc.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 4)
        pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        ts = getattr(self, "team_safety", None)
        if ts is not None and ts.enabled:
            # team: no cross-body reach -> the hand-over spot must be reachable by BOTH hands: the middle band.
            # Denser grid there (the layout grid has few columns near y = 0).
            band = float(rc.get("mid_band", 0.05))
            px = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 6)          # pelvis x (forward) over the region
            cand = [np.array([x, y]) for x in px for y in np.linspace(-band, band, 5)]  # pelvis y around 0
            keep = [p for p in cand if np.linalg.norm(p - det.top[:2]) > 0.06
                    and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        keep = keep[: int(rc.get("max_spots", 6))]
        if getattr(self, "insert_holes", None) and rc.get("insert_spots"):
            # team task 3: known-good hand-over spots for the lid arm first (mirrored for a right lid bin)
            sg = 1.0 if b[1] > 0 else -1.0
            fixed = [np.array([x, sg * y]) for x, y in rc["insert_spots"]]
            fixed = [p for p in fixed if np.linalg.norm(p - det.top[:2]) > 0.06
                     and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
            keep = fixed + keep
        return keep

    def plan_relocation(self, det, bin_name, arm_q, hand_q, obstacles=()):
        """Cross-body fallback: pick `det` with an arm that reaches it and put it down at a counter spot from which
        some arm has a full plan to `bin_name` (checked from home). Returns a Plan with target_bin 'counter'."""
        from dataclasses import replace
        last = Plan(False, color=det.color, reason="no relocation spot")
        H = self.height
        for spot in self.relocation_spots(det, bin_name, obstacles):
            d2 = replace(det, top=np.r_[spot, self.scene.counter_z + H])
            by = self.bin_xy(bin_name)[1]
            fin_side = "left" if (by if abs(by) > 0.1 else spot[1]) > 0 else "right"
            skey = (getattr(det, "shape", None), fin_side, bin_name, round(float(spot[0]), 3), round(float(spot[1]), 3))
            spot_ok = self.__dict__.setdefault("_spot_ok", {})
            if skey in spot_ok:   # team speed: this spot was already shown finishable for this shape + arm
                fin = spot_ok[skey]
            else:
                fin = self._plan_side_hints(d2, fin_side, bin_name, self.home.copy(), self.post["open"].copy())
                if fin.ok:
                    spot_ok[skey] = fin
            if not fin.ok:
                last.reason = f"no {fin_side} plan from spot ({spot[0]:+.2f},{spot[1]:+.2f}): {fin.reason}"
                if os.environ.get("AALTO_PLAN_DEBUG"):
                    print(f"   [relocate] {last.reason[:300]}", flush=True)
                continue
            side = "right" if fin_side == "left" else "left"
            p = self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=spot)
            if p.ok:
                p.reason = f"relocate for the {fin_side} arm"
                p.handover_fin = fin   # team: the other arm's full plan from this spot (from home), reused after
                return p
            last = p
            last.reason = f"{side} cannot place at ({spot[0]:+.2f},{spot[1]:+.2f}): {p.reason}"
            if os.environ.get("AALTO_PLAN_DEBUG"):
                print(f"   [relocate] {last.reason[:300]}", flush=True)
        return last
    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        """team: task 3, lid on the bin FLOOR: a hole near the bin walls (square / circle corners, 32 mm from the
        walls) puts the outer finger below the rim at the normal release height -> retry with the release raised in
        steps (grasp.insert_rim_raise) until the hand clears the rim; the object drops the rest into the hole."""
        hole = None if kw.get("place_xy") is not None else \
            (getattr(self, "insert_holes", None) or {}).get(getattr(det, "shape", None))
        raises = list(self.cfg["grasp"].get("insert_rim_raise") or [])
        rc = self.__dict__.get("_raise_ok", {}).get((getattr(det, "shape", None), side), 0.0)
        p = None
        if hole is None or "dz" not in hole or not raises or rc == 0.0:
            p = self._plan_side_hints_once(det, side, bin_name, arm_q, hand_q, **kw)
            if p.ok or hole is None or "dz" not in hole or not raises or "rim" not in (p.reason or ""):
                return p
        dz0 = hole["dz"]
        # team speed: start from the raise that last worked for this hole + arm (the rim geometry does not change)
        cache = self.__dict__.setdefault("_raise_ok", {})
        key = (det.shape, side)
        if key in cache:
            raises = [r for r in raises if r >= cache[key]]
        try:
            for r in raises:
                hole["dz"] = dz0 + r
                p = self._plan_side_hints_once(det, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    cache[key] = r
                    p.waypoints["release_raise"] = np.r_[r, 0.0, 0.0]
                    return p
                if "rim" not in (p.reason or ""):
                    break
            return p
        finally:
            hole["dz"] = dz0

    def _plan_side_hints_once(self, det, side, bin_name, arm_q, hand_q, **kw):
        """_plan_side with each finger direction of grasp.finger_hints in turn (fingers forward can reach over the
        centre bin's wall when the object is close to it; then fingers inward/sideways)."""
        g = self.cfg["grasp"]
        hints = g.get("finger_hints") or [g.get("finger_hint", [1.0, 0.0, 0.0])]
        hints = [np.asarray(h, float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0) for h in hints]
        hints = [h / np.linalg.norm(h) for h in hints]
        saved, saved_w, saved_off, first = self.kin.hint, self.kin.w_hint, self.kin.offset[side].copy(), None
        saved_ax = self.kin.hint_axis[side].copy()
        sg = g.get("shape_grasp") or {}
        sd = g.get("side_grasp") or {}
        if sd.get("enabled"):
            # team: SIDE grasp. Approach axis tilted tilt_deg from straight down, towards the object as seen from the
            # pelvis (the hand comes in from the robot's side); fingers (pinch axis) horizontal, across that direction.
            saved_app = self.kin.approach[side].copy()
            u = np.r_[det.top[:2], 0.0]
            u /= max(np.linalg.norm(u), 1e-9)
            perp = np.array([-u[1], u[0], 0.0])
            first = None
            try:
                for tdeg in sd.get("tilt_deg", [70, 60, 50]):
                    t = np.radians(float(tdeg))
                    a = np.cos(t) * np.array([0.0, 0.0, -1.0]) + np.sin(t) * u
                    self.kin.approach[side] = a / np.linalg.norm(a)
                    for h in (perp, -perp):
                        self.kin.hint = h
                        p = self._plan_side(det, side, bin_name, arm_q, hand_q, approach_vec=self.kin.approach[side],
                                            **kw)
                        if p.ok:
                            p.shape = f"side{tdeg:.0f}"
                            return p
                        first = first or p
                return first
            finally:
                self.kin.approach[side] = saved_app
                self.kin.hint = saved
        prof = (g.get("shape_profiles") or {}).get(getattr(det, "shape", None) or "", None)
        try:
            # team: shape-aware pinch. The thumb-finger pinch axis is the finger direction (palm x), so align it
            # with the object's flats: cube = a face normal, triangle = flat -> apex. Most-preferred direction first
            # (closest to the generic hints); then the generic, shape-agnostic hints as a fallback.
            if sg.get("enabled") and getattr(det, "grasp_dirs", None) and det.shape in ("cube", "triangle", "cylinder"):
                dirs = [np.r_[np.asarray(d, float), 0.0] for d in det.grasp_dirs]
                pref = lambda d: min(np.degrees(np.arccos(np.clip(d @ h, -1, 1))) for h in hints)  # noqa: E731
                dirs = [d for d in sorted(dirs, key=pref) if pref(d) <= float(sg.get("max_off_hint_deg", 100))]
                self.kin.w_hint = float(sg.get("hint_weight", 2.0))
                self.kin.hint_signed = det.shape == "triangle"
                mir = np.array([1.0, -1.0, 1.0]) if side == "right" else np.ones(3)
                if prof and prof.get("enabled", True):
                    # team: grip profile learned from the real demos of this shape: grasp point + pinch axis (palm
                    # frame, from the demo fingertip geometry at contact), postures by name (hand.<name>)
                    self.kin.offset[side] = np.asarray(prof["point_in_palm"], float) * mir
                    ax = np.asarray(prof["pinch_axis_in_palm"], float) * mir
                    ax[1] = 0.0  # keep it horizontal when the approach axis (palm -y) points down
                    self.kin.hint_axis[side] = ax / np.linalg.norm(ax)
                elif det.shape == "triangle":
                    self.kin.offset[side] = saved_off + np.asarray(sg.get("triangle_palm_shift", [0, 0, 0]), float) * mir
                for d in dirs:
                    self.kin.hint = d
                    p = self._plan_side(det, side, bin_name, arm_q, hand_q, pinch_dir=d,
                                        profile=prof if prof and prof.get("enabled", True) else None, **kw)
                    if p.ok:
                        return p
                    first = first or p
                self.kin.w_hint, self.kin.offset[side] = saved_w, saved_off.copy()
                self.kin.hint_axis[side] = saved_ax.copy()
            for h in hints:
                self.kin.hint = h
                p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    return p
                first = first or p
            return first
        finally:
            self.kin.hint, self.kin.w_hint, self.kin.offset[side] = saved, saved_w, saved_off
            self.kin.hint_axis[side] = saved_ax
            self.kin.hint_signed = False

    def nudge_plan(self, side, arm_q, hand_q, start_xy, end_xy, z):
        """Team: push with the closed hand along start_xy -> end_xy at height z (pelvis frame): pre (8 cm above
        start) -> down -> push -> up 8 cm -> home. Same counter / bin / team-safety / safety-filter checks as picks
        (the push itself may come within 3 mm of the counter). Returns a Plan (ok or with a reason)."""
        g, mo = self.cfg["grasp"], self.cfg["motion"]
        plan = Plan(False, side=side, color="nudge", target_bin="nudge")
        sl = ArmKinematics.sl(side)
        h_cls = np.asarray(hand_q, float).copy(); h_cls[sl] = self.post["closed"][sl]
        h_open = np.asarray(hand_q, float).copy(); h_open[sl] = self.post["open"][sl]
        s0, s1 = np.r_[start_xy, z], np.r_[end_xy, z]
        sc_ = self.scene
        z_carry = sc_.counter_z + max(g["carry_height"], float(sc_.cfg["bins"]["wall_height"]) + self.height + 0.03)
        pre, post_ = np.r_[start_xy, max(z + 0.08, z_carry)], np.r_[end_xy, max(z + 0.08, z_carry)]  # like picks
        q_pre, ok = self._ik(side, pre, np.asarray(arm_q, float), "nudge_pre", plan)
        if not ok:
            plan.reason = "IK nudge pre"
            return plan
        down = self._cartesian(side, pre, s0, q_pre, "nudge_down", plan)
        push = down and self._cartesian(side, s0, s1, down[-1], "nudge_push", plan)
        up = push and self._cartesian(side, s1, post_, push[-1], "nudge_up", plan)
        if not up:
            plan.reason = plan.reason or "IK nudge path"
            return plan
        vmax, amax = mo["arm_max_vel"], mo["arm_max_acc"]
        home = np.asarray(arm_q, float).copy(); home[sl] = self.home[sl]
        parts, hparts, phases, cur, k = [], [], [], np.asarray(arm_q, float), 0
        cur_h = np.asarray(hand_q, float)
        for name, Qs, hv in (("nudge_to_pre", [q_pre], h_cls), ("nudge_down", down, h_cls),
                             ("nudge_push", push, h_cls), ("nudge_up", up, h_cls), ("home", [home], h_open)):
            vm = vmax * (0.4 if name == "nudge_push" else 1.0)  # push slowly
            a = time_scale([cur] + list(Qs), vm, amax, self.dt)
            hr = ramp(cur_h, hv, mo["hand_max_vel"], self.dt)  # hand moves at its speed limit, never jumps
            n = max(len(a), len(hr))
            if n == 0:
                continue
            if len(a) < n:
                a = np.vstack([a, np.repeat((a[-1] if len(a) else cur)[None], n - len(a), 0)])
            h = np.repeat(np.asarray(hv, float)[None], n, 0)
            h[:len(hr)] = hr
            parts.append(a); hparts.append(h); phases.append((name, k, k + n)); k += n
            cur, cur_h = a[-1], h[-1]
        plan.arm, plan.hand, plan.phases = np.vstack(parts), np.vstack(hparts), phases
        n_up = next(e - s_ for nm, s_, e in phases if nm == "nudge_up") if any(nm == "nudge_up" for nm, _, _ in phases) else 0
        for i in range(0, len(plan.arm), 3):
            name = next(n for n, s_, e in phases if s_ <= i < e)
            # near the object (down, push, the first half of up) the fist may come within 3 mm of the counter
            near = name in ("nudge_down", "nudge_push") or (
                name == "nudge_up" and i < next(s_ for nm, s_, e in phases if nm == "nudge_up") + n_up // 2)
            need = 0.003 if near else mo["counter_clearance"]
            if self.clearance(side, plan.arm[i], plan.hand[i]) < need - 2e-3:
                plan.reason = f"counter clearance during {name}"
                return plan
            if self.bin_clearance(side, plan.arm[i], plan.hand[i]) < 0.003:
                plan.reason = f"bin rim during {name}"
                return plan
        ts = getattr(self, "team_safety", None)
        if ts is not None:
            bad = ts.check_traj(plan.arm, plan.hand, start=(arm_q, hand_q))
            if bad:
                plan.reason = f"team safety: {bad[1]}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips"
            return plan
        plan.ok = True
        return plan

    def quick_pinch_quality(self, det_xy, z, dirs, arm_q, side, signed=True):
        """Team: fast check (grasp IK only, no full plan) of pinch directions for one hand: best (score, dir, wrist
        margin, finger err) with the team safety layer at the grasp pose, else None."""
        sg = self.cfg["grasp"].get("shape_grasp") or {}
        sl = ArmKinematics.sl(side)
        saved = (self.kin.hint, self.kin.w_hint, self.kin.hint_signed)
        best = None
        try:
            self.kin.w_hint, self.kin.hint_signed = float(sg.get("hint_weight", 2.0)), signed
            lo, hi = np.asarray(self.spec.arm_lower)[sl], np.asarray(self.spec.arm_upper)[sl]
            for d2 in dirs:
                d = np.r_[np.asarray(d2, float), 0.0]
                self.kin.hint = d
                q, ok, err, tilt = self.kin.solve(side, np.r_[det_xy, z], np.asarray(arm_q, float), top_down=True,
                                                  extra_seeds=[self.home[sl]])
                if not ok:
                    continue
                fe = self.finger_err_deg(side, q, d)
                if fe > float(sg.get("max_finger_err_deg", 20.0)):
                    continue
                ts = getattr(self, "team_safety", None)
                if ts is not None and ts.check(q, None):
                    continue
                wm = float(np.min(np.minimum(q[sl][4:] - lo[4:], hi[4:] - q[sl][4:])))
                sc = fe + 100.0 * max(0.0, 0.35 - wm)
                if best is None or sc < best[0]:
                    best = (sc, d[:2], wm, fe)
        finally:
            self.kin.hint, self.kin.w_hint, self.kin.hint_signed = saved
        return best

    def best_pinch_plan(self, det, arm_q, hand_q, bin_name, place_xy=None, log=None):
        """Team: hand choice from the object's orientation. Plans every (hand, pinch direction) combination of a
        shape-classified object (each candidate passes the team safety layer + safety dry run inside _plan_side) and
        returns the ok plan with the best accuracy score (lower = better):
            finger_err_deg + tilt_weight * grasp tilt_deg + wrist_weight * max(0, margin_rad - wrist limit margin)
        Wrist margin = distance (rad) of the grasp pose's wrist joints from their URDF limits. Falls back to plan()."""
        g = self.cfg["grasp"]
        sg, hs = g.get("shape_grasp") or {}, g.get("hand_select") or {}
        if not (sg.get("enabled") and getattr(det, "grasp_dirs", None) and det.shape in ("cube", "triangle")):
            return None, []
        prof = (g.get("shape_profiles") or {}).get(det.shape, None)
        prof = prof if prof and prof.get("enabled", True) else None
        cands = []
        for side in ("left", "right"):
            sl = ArmKinematics.sl(side)
            saved = (self.kin.hint, self.kin.w_hint, self.kin.offset[side].copy(), self.kin.hint_axis[side].copy())
            try:
                self.kin.hint_signed = det.shape == "triangle"
                self.kin.w_hint = float(sg.get("hint_weight", 2.0))
                mir = np.array([1.0, -1.0, 1.0]) if side == "right" else np.ones(3)
                if prof:
                    self.kin.offset[side] = np.asarray(prof["point_in_palm"], float) * mir
                    ax = np.asarray(prof["pinch_axis_in_palm"], float) * mir
                    ax[1] = 0.0
                    self.kin.hint_axis[side] = ax / np.linalg.norm(ax)
                for d2 in det.grasp_dirs:
                    d = np.r_[np.asarray(d2, float), 0.0]
                    self.kin.hint = d
                    p = self._plan_side(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float),
                                        place_xy=place_xy, pinch_dir=d, profile=prof)
                    if not p.ok:
                        cands.append({"side": side, "dir_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                                      "ok": False, "why": p.reason[:80]})
                        continue
                    i = {n: (s, e) for n, s, e in p.phases}["close"][0]
                    q7 = p.arm[i][sl]
                    lo, hi = np.asarray(self.spec.arm_lower)[sl], np.asarray(self.spec.arm_upper)[sl]
                    wm = float(np.min(np.minimum(q7[4:] - lo[4:], hi[4:] - q7[4:])))  # wrist roll/pitch/yaw
                    tilt = float(p.ik.get("grasp", (True, 0, 0))[2])
                    score = (float(p.finger_err_deg or 0.0) + float(hs.get("tilt_weight", 0.5)) * tilt
                             + float(hs.get("wrist_weight", 100.0)) * max(0.0, float(hs.get("wrist_margin", 0.35)) - wm))
                    cands.append({"side": side, "dir_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                                  "ok": True, "score": round(score, 1), "finger_err": p.finger_err_deg,
                                  "tilt": round(tilt, 1), "wrist_margin": round(wm, 2), "dur": round(p.duration, 1),
                                  "plan": p})
                    if score <= float(hs.get("good_enough_score", 1.0)):
                        break  # team speed-up: an excellent plan (aligned, comfortable wrist): stop searching
            finally:
                self.kin.hint, self.kin.w_hint, self.kin.offset[side], self.kin.hint_axis[side] = saved
                self.kin.hint_signed = False
            if any(c["ok"] and c["score"] <= float(hs.get("good_enough_score", 1.0)) for c in cands):
                break  # ... and do not plan the other hand either
        ok = [c for c in cands if c["ok"]]
        if log:
            log("hand_select: " + "; ".join(f"{c['side']} {c['dir_deg']:+.0f}deg " +
                                           (f"score {c['score']}" if c["ok"] else f"X {c['why']}") for c in cands))
        if not ok:
            return None, cands
        best = min(ok, key=lambda c: c["score"])
        return best["plan"], [{k: v for k, v in c.items() if k != "plan"} for c in cands]

    def finger_err_deg(self, side, arm_q, pinch_dir):
        """Team: angle (deg) between the horizontal finger direction (palm x) at arm_q and the wanted pinch axis.
        A cube pinch is symmetric (f or -f both fine); a triangle pinch is not (fingers must point at the apex)."""
        self.kin._fk(np.asarray(arm_q, float))
        f = self.kin.data.oMf[self.kin.palm[side]].rotation @ self.kin.hint_axis[side]
        f[2] = 0.0
        n = np.linalg.norm(f)
        if n < 1e-6:
            return 180.0
        c = float(np.clip((f / n) @ np.asarray(pinch_dir, float)[:3], -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None, pinch_dir=None, profile=None,
                   approach_vec=None):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter")
        sl = ArmKinematics.sl(side)
        H, post = self.height, self.post
        shape = det.shape if pinch_dir is not None else None
        plan.shape = shape + ("/demo" if profile else "") if shape else None
        pre_name = (profile or {}).get("preshape", "preshape")
        hand_pre = hand_q.copy(); hand_pre[sl] = post[pre_name][sl]
        cls_name = f"closed_{shape}" if shape and f"closed_{shape}" in post else "closed"
        if profile:
            cls_name = profile.get("closed", cls_name)
        hand_cls = hand_q.copy(); hand_cls[sl] = post[cls_name][sl]
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]

        # grasp height: object mid-height, raised until the preshaped fingers clear the counter
        xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
        if pinch_dir is not None:  # triangle: the pinch middle is between the flat and the apex, not the centroid
            xy = xy + float(getattr(det, "centroid_to_pinch", 0.0)) * np.asarray(pinch_dir, float)[:2]
        # team: per-shape grasp depth (prisms are picked last, on an emptier table: grip lower, closer to the counter)
        _sh = getattr(det, "shape", None)
        hf = float((g.get("shape_height_frac") or {}).get(_sh, g["height_frac"]))
        ftc = float((g.get("shape_fingertip_clearance") or {}).get(_sh, g["fingertip_clearance"]))
        z = zc + hf * H
        if g.get("depth_grasp_height") and getattr(det, "method", "") == "depth":
            # team: grip at height_frac of the MEASURED object height (top face from head depth), not scene.yaml's
            Hm = float(np.clip(det.top[2] - zc, 0.5 * H, 1.5 * H))
            z = zc + hf * Hm
            plan.waypoints["measured_height"] = np.r_[det.top[:2], Hm]
        dft = g.get("grasp_depth_from_top")
        if dft:
            # team: grasp point a fixed depth BELOW the object's top (measured by head depth when available); the
            # fingertip clearance loop below may still raise it so the fingertips stay above the table
            top = zc + (Hm if g.get("depth_grasp_height") and getattr(det, "method", "") == "depth" else H)
            z = max(zc + 0.005, top - float(dft))
        for _ in range(4):
            qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
            if not ok:
                plan.reason = f"IK grasp {side}"
                return plan
            if pinch_dir is not None:
                err = self.finger_err_deg(side, qg, pinch_dir)
                if shape == "cube":
                    err = min(err, 180.0 - err)
                plan.finger_err_deg = round(err, 1)
                tol = float((g.get("shape_grasp") or {}).get("max_finger_err_deg", 20.0))
                if err > tol:
                    plan.reason = f"{shape} pinch axis off by {err:.0f} deg (> {tol:.0f})"
                    return plan
            c = self.clearance(side, qg, hand_pre, fingers_only=True)
            if c >= ftc - 1e-3:
                break
            z += ftc - c
        else:
            plan.reason = "fingers hit the counter"
            return plan
        if z > zc + H - 0.008:
            plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
            return plan

        wall = float(sc.cfg["bins"]["wall_height"])
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        z_drop = zc + wall + H / 2 + g["drop_above_rim"]
        bxy = self.drop_xy(bin_name)
        far = (getattr(self, "far_drop", None) or {}).get(bin_name)
        if far is not None:   # team: far boxes (real setup): the deepest release point the arms reach (policy)
            bxy = np.asarray(far["xy"], float)
            z_drop = zc + wall + H / 2 + far["above"]
            z_carry = max(z_carry, z_drop + 0.02)
        hole = (getattr(self, "insert_holes", None) or {}).get(getattr(det, "shape", None)) if place_xy is None else None
        if hole is not None:  # team rung 3: release above the matching lid hole, just over the lid
            bxy = np.asarray(hole["xy"], float)
            z_drop = float(hole["z"]) + H / 2 + float(hole.get("dz", g.get("insert_drop_above_lid", 0.02)))
        if place_xy is not None:  # relocation: put it down on the counter (release just above the grasp height)
            bxy, z_drop = np.asarray(place_xy, float), z + 0.005
        P = {"pregrasp": np.r_[xy, max(z + g["approach_height"], z_carry)], "grasp": np.r_[xy, z], "lift": np.r_[xy, z_carry],
             "above_bin": np.r_[bxy, z_carry], "drop": np.r_[bxy, z_drop], "retreat": np.r_[bxy, z_drop + g["retreat_up"]]}
        ao = g.get("aim_offset") or {}
        if ao and place_xy is None:   # team (real robot): aim where the sagging arm actually lands (measured)
            for k in ("pregrasp", "grasp", "lift"):
                P[k] = P[k] + np.asarray(ao.get("grasp", [0, 0, 0]), float) * ([1, -1, 1] if side == "right" else 1)
            for k in ("above_bin", "drop", "retreat"):
                P[k] = P[k] + np.asarray(ao.get("drop", [0, 0, 0]), float) * ([1, -1, 1] if side == "right" else 1)
        if approach_vec is not None:  # team side grasp: come in along the approach axis, not from straight above
            P["pregrasp"] = np.r_[xy, z] - np.asarray(approach_vec, float) * float(
                (g.get("side_grasp") or {}).get("approach_dist", 0.10))

        q_pre, ok = self._ik(side, P["pregrasp"], qg, "pregrasp", plan)  # same IK branch as the grasp
        if not ok:
            plan.reason = f"IK pregrasp {side}"
            return plan
        plan.waypoints["grasp"] = np.r_[xy, z]
        down = self._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
        up = down and self._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
        if hole is None and pinch_dir is not None and g.get("free_wrist_on_carry", True):
            # team: sorting -> the object's yaw at the bin does not matter. Keep the shape-aligned orientation for the
            # grasp + lift only; carry / lower / retreat with the generic, weak finger hint (free wrist), or the
            # held orientation can put the bin out of reach (all prism wedge plans failed on the carry before).
            gh = np.asarray((g.get("finger_hints") or [g.get("finger_hint", [1.0, 0.0, 0.0])])[0], float)
            gh = gh * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
            self.kin.hint = gh / np.linalg.norm(gh)
            self.kin.w_hint = float(g.get("finger_hint_weight", 0.3))
            self.kin.hint_signed = False
            self.kin.hint_axis[side] = np.array([1.0, 0.0, 0.0])
        cand_dirs = [None]
        if hole is not None and hole.get("dirs") and pinch_dir is not None:
            # turn the object to the hole on the way: pinch direction (object flat normal / flat->apex) onto one of
            # the hole's matching directions. Smallest wrist turn first; team: the others too (the nearest one can be
            # out of reach with the wedge grip's offset palm point, e.g. the triangle hole for the left arm)
            d_now = np.asarray(pinch_dir, float)[:2]
            cand_dirs = sorted(hole["dirs"], key=lambda v: -float(np.dot(v, d_now)))
        retreat = None
        for d_drop in cand_dirs:
            if d_drop is not None:
                self.kin.hint = np.r_[d_drop, 0.0]
                plan.waypoints["hole_dir"] = np.r_[d_drop, 0.0]
                plan.reason = ""
            # team: the joint-space carry must keep the object over the bin RIM (floor lid: z_drop is below the rim,
            # and the swing used to cut through the wall); only the straight "lower" goes down into the bin
            z_rim = zc + wall + H / 2 + g["drop_above_rim"]
            carry = up and (self._cartesian(side, P["lift"], P["above_bin"], up[-1], "carry", plan)
                            or self._joint_carry(side, up[-1], P["lift"], P["above_bin"], z_rim,
                                                 "carry", plan))
            lower = carry and self._cartesian(side, P["above_bin"], P["drop"], carry[-1], "lower", plan)
            retreat = lower and self._cartesian(side, P["drop"], P["retreat"], lower[-1], "retreat", plan)
            if retreat or not up:
                break
        if not retreat:
            plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            return plan
        plan.waypoints["drop"] = P["drop"]

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def move(name, Qs, hand_to=None):
            nonlocal cur_a, cur_h
            a = time_scale([cur_a] + list(Qs), vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:  # hand moves during the arm move (ends no later than the arm)
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            h = ramp(cur_h, to, hv, self.dt)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        move("to_pregrasp", [q_pre], hand_to=hand_pre)
        move("descend", down)
        if float(mo.get("pre_close_hold_s", 0.0)) > 0:   # team (real robot): let the lagging arm reach the grasp
            add("close_wait", np.repeat(cur_a[None], int(round(float(mo["pre_close_hold_s"]) / self.dt)), 0),
                np.repeat(cur_h[None], int(round(float(mo["pre_close_hold_s"]) / self.dt)), 0))
        hand("close", hand_cls, mo["close_hold_s"])
        move("lift", up)
        move("carry", carry + lower)
        if float(mo.get("pre_release_hold_s", 0.0)) > 0:   # team (real robot): arm lagged 5-13 cm at the drop
            n_w = int(round(float(mo["pre_release_hold_s"]) / self.dt))
            add("open_wait", np.repeat(cur_a[None], n_w, 0), np.repeat(cur_h[None], n_w, 0))
        if cls_name == "closed_wedge":
            # team: V-wedge release. The prism is clamped between the SIDES of the hanging fingers; straightening them
            # just rotates it along (no finger spread joint). Undo the wedge instead: thumb back first (stop pushing),
            # then slide the hand towards the prism's apex so the finger walls reach its narrow part -> it drops.
            th = slice(sl.start, sl.start + 3)                       # thumb joints of this hand (DDS order)
            h_thumb = cur_h.copy(); h_thumb[th] = post["preshape"][th]
            hand("release_thumb", h_thumb, 0.15)
            self.kin._fk(cur_a)
            f = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0].copy()
            f[2] = 0.0
            f /= max(np.linalg.norm(f), 1e-9)
            slide_p = P["drop"] + np.r_[f[:2], 0.0] * float(g.get("wedge_release_slide", 0.03))
            slide = self._cartesian(side, P["drop"], slide_p, cur_a, "release_slide", plan)
            if slide:
                move("release_slide", slide)
                r2 = self._cartesian(side, slide_p, slide_p + [0, 0, g["retreat_up"]], slide[-1], "retreat", plan)
                if r2:
                    retreat = r2
        hand("open", hand_rel, mo["open_hold_s"])  # widest posture: `open` alone left the cube wedged in the sim
        amp = float(g.get("release_shake", 0.0) or 0.0)
        if amp > 0:  # team: wrist wiggle over the bin to throw off an object wedged between index and middle
            spec = self.kin.spec
            lo = np.asarray(spec.arm_lower)[sl] + 0.1
            hi = np.asarray(spec.arm_upper)[sl] - 0.1
            q0 = cur_a.copy(); seq = []
            for sgn in (1, -1, 1, -1):
                q = q0.copy()
                q[sl] = np.clip(q[sl] + sgn * amp * np.array([0, 0, 0, 0, 1.0, 0.6, 0]), lo, hi)
                seq.append(q)
            move("shake", seq + [q0])
        move("retreat", retreat)
        move("home", [home], hand_to=hand_open)
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        grasp_ticks = np.zeros(len(plan.arm), bool)
        near = ("descend", "close_wait", "close", "lift") + (("carry", "open_wait", "open", "retreat", "release_thumb",
                                                               "release_slide")
                                                if place_xy is not None else ())
        for name, s, e in phases:
            if name in near:
                grasp_ticks[s:e] = True
        closing = np.zeros(len(plan.arm), bool)
        for name, s, e in phases:
            closing[s:e] |= name == "close" or (place_xy is not None and name == "open")
        for k in range(0, len(plan.arm), 3):
            # closing: the thumb swings through its lowest point; touching the mat lightly is acceptable
            need = 0.003 if closing[k] else ftc if grasp_ticks[k] else mo["counter_clearance"]
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                # the object itself sits between the fingers during descend/lift; only the counter matters here
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
                return plan
        drop = np.zeros(len(plan.arm), bool)
        for name, s_, e in phases:
            drop[s_:e] |= name in ("open", "release_thumb", "release_slide")
        for k in range(0, len(plan.arm), 3):
            if drop[k]:
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                name = next(n for n, s_, e in phases if s_ <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        ts = getattr(self, "team_safety", None)
        if ts is not None:  # team: self-collision (arm-arm, arm-torso) + no cross-body reach, on the whole plan
            bad = ts.check_traj(plan.arm, plan.hand, start=(arm_q, hand_q))
            if bad:
                name = next((n for n, s_, e in phases if s_ <= bad[0] < e), "?")
                plan.reason = f"team safety during {name}: {bad[1]}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan

    # ------------------------------------------------------------------ team: feedback-layer segments
    def _segment_ok(self, side, arm_q, hand_q, a, h, near_counter=False):
        """Counter + bin-rim clearance and a safety dry run for a short segment. Returns a reason or None."""
        mo = self.cfg["motion"]
        need = 0.003 if near_counter else mo["counter_clearance"]
        for k in range(0, len(a), 3):
            if self.clearance(side, a[k], h[k]) < need - 2e-3:
                return "counter clearance"
            if self.bin_clearance(side, a[k], h[k]) < 0.003:
                return "bin rim clearance"
        ts = getattr(self, "team_safety", None)
        if ts is not None:
            bad = ts.check_traj(a, h, start=(arm_q, hand_q))
            if bad:
                return f"team safety: {bad[1]}"
        clips = self.dry_run_safety(arm_q, hand_q, a, h)
        return f"{clips} safety clips" if clips else None

    def place_plan(self, side, arm_q, hand_q, target_xy, z_drop, bin_name, color="", shape=None, hint=None,
                   signed=False):
        """Team: re-plan the place part from the current (lifted, holding) pose: carry -> lower -> open -> retreat ->
        home, releasing with the GRASP POINT over target_xy at z_drop. hint = finger direction to hold while carrying
        (lid holes). Same counter / bin / team-safety / safety-filter checks as a full plan. Returns a Plan."""
        g, mo = self.cfg["grasp"], self.cfg["motion"]
        sc, zc, H = self.scene, self.scene.counter_z, self.height
        plan = Plan(False, side=side, color=color, target_bin=bin_name)
        plan.shape = shape
        sl = ArmKinematics.sl(side)
        arm_q, hand_q = np.asarray(arm_q, float), np.asarray(hand_q, float)
        p_now = self.kin.grasp_pose(side, arm_q)[0]
        wall = float(sc.cfg["bins"]["wall_height"])
        z_carry = max(p_now[2], zc + max(g["carry_height"], wall + H + 0.03))
        above, drop = np.r_[target_xy, z_carry], np.r_[target_xy, z_drop]
        retreat_p = drop + [0, 0, g["retreat_up"]]
        saved = (self.kin.hint, self.kin.w_hint, self.kin.hint_signed)
        try:
            if hint is not None:
                self.kin.hint = np.r_[np.asarray(hint, float)[:2], 0.0]
                self.kin.w_hint = float((g.get("shape_grasp") or {}).get("hint_weight", 2.0))
                self.kin.hint_signed = bool(signed)
            carry = (self._cartesian(side, p_now, above, arm_q, "carry", plan)
                     or self._joint_carry(side, arm_q, p_now, above, z_drop, "carry", plan))
            lower = carry and self._cartesian(side, above, drop, carry[-1], "lower", plan)
            ret = lower and self._cartesian(side, drop, retreat_p, lower[-1], "retreat", plan)
        finally:
            self.kin.hint, self.kin.w_hint, self.kin.hint_signed = saved
        if not ret:
            plan.reason = plan.reason or "IK place path"
            return plan
        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        h_rel = hand_q.copy(); h_rel[sl] = self.post[g.get("release_posture", "preshape")][sl]
        h_open = hand_q.copy(); h_open[sl] = self.post["open"][sl]
        parts, hparts, phases, cur_a, cur_h, k = [], [], [], arm_q, hand_q, 0

        def add(name, a, h):
            nonlocal cur_a, cur_h, k
            if len(a):
                parts.append(a); hparts.append(h); phases.append((name, k, k + len(a))); k += len(a)
                cur_a, cur_h = a[-1].copy(), h[-1].copy()

        def move(name, Qs, hand_to=None):
            a = time_scale([cur_a] + list(Qs), vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
            add(name, a, h)

        move("carry", carry + lower)
        hr = ramp(cur_h, h_rel, hv, self.dt)
        hr = np.vstack([hr, np.repeat(h_rel[None], int(round(mo["open_hold_s"] / self.dt)), 0)])
        add("open", np.repeat(cur_a[None], len(hr), 0), hr)
        move("retreat", ret)
        move("home", [home], hand_to=h_open)
        plan.arm, plan.hand, plan.phases = np.vstack(parts), np.vstack(hparts), phases
        plan.waypoints["drop"] = drop
        for i in range(0, len(plan.arm), 3):
            name = next(n for n, s_, e in phases if s_ <= i < e)
            if self.clearance(side, plan.arm[i], plan.hand[i]) < mo["counter_clearance"] - 2e-3:
                plan.reason = f"counter clearance during {name}"
                return plan
            if name != "open" and self.bin_clearance(side, plan.arm[i], plan.hand[i]) < 0.003:
                plan.reason = f"bin rim during {name}"
                return plan
        ts = getattr(self, "team_safety", None)
        if ts is not None:
            bad = ts.check_traj(plan.arm, plan.hand, start=(arm_q, hand_q))
            if bad:
                plan.reason = f"team safety: {bad[1]}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips"
            return plan
        plan.ok = True
        return plan

    def abort_segment(self, side, arm_q, hand_q):
        """Grasp failed (seen at the lift): open the hand where it is, then joint-space home. (a, h) or None."""
        g, mo, sl = self.cfg["grasp"], self.cfg["motion"], ArmKinematics.sl(side)
        h_rel = np.asarray(hand_q, float).copy(); h_rel[sl] = self.post[g.get("release_posture", "preshape")][sl]
        h1 = ramp(hand_q, h_rel, mo["hand_max_vel"], self.dt)
        h1 = np.vstack([h1, np.repeat(h_rel[None], int(round(0.15 / self.dt)), 0)]) if len(h1) else h_rel[None]
        a1 = np.repeat(np.asarray(arm_q, float)[None], len(h1), 0)
        home = np.asarray(arm_q, float).copy(); home[sl] = self.home[sl]
        h_open = h_rel.copy(); h_open[sl] = self.post["open"][sl]
        a2, h2 = self.go_to(arm_q, h_rel, home, h_open)
        a, h = np.vstack([a1, a2]), np.vstack([h1, h2])
        why = self._segment_ok(side, arm_q, hand_q, a, h)
        return None if why else (a, h)

    def shake_segment(self, side, arm_q, hand_q, amp=None):
        """Object still in the hand after the release (seen over the bin): wrist roll/pitch wiggle with the hand
        in the release posture, back to the same pose. (a, h) or None."""
        g, mo, sl = self.cfg["grasp"], self.cfg["motion"], ArmKinematics.sl(side)
        amp = float(amp if amp is not None else (g.get("feedback") or {}).get("shake_amp", 0.3))
        q0 = np.asarray(arm_q, float).copy()
        lo = np.asarray(self.spec.arm_lower)[sl] + 0.1
        hi = np.asarray(self.spec.arm_upper)[sl] - 0.1
        seq = [q0]
        for sgn in (1, -1, 1, -1):
            q = q0.copy()
            q[sl] = np.clip(q0[sl] + sgn * amp * np.array([0, 0, 0, 0, 1.0, 0.6, 0]), lo, hi)
            seq.append(q)
        seq.append(q0)
        a = time_scale(seq, mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h_rel = np.asarray(hand_q, float).copy(); h_rel[sl] = self.post[g.get("release_posture", "preshape")][sl]
        h = np.repeat(h_rel[None], len(a), 0)
        hold = int(round(0.3 / self.dt))
        a = np.vstack([a, np.repeat(q0[None], hold, 0)])
        h = np.vstack([h, np.repeat(h_rel[None], hold, 0)])
        why = self._segment_ok(side, arm_q, hand_q, a, h)
        return None if why else (a, h)

    def dry_run_safety(self, arm_q, hand_q, arm, hand):
        f = SafetyFilter(self.spec, self.safety)
        f.reset(arm_q, hand_q, 0.0)
        for k in range(len(arm)):
            f.filter({"arm_q_target": arm[k], "hand_q_target": hand[k]}, (k + 1) * self.dt)
        return f.clipped_steps

    def go_to(self, arm_from, hand_from, arm_to, hand_to):
        """Joint-space move (both arms + hands) at the configured speeds, e.g. start pose -> home."""
        mo = self.cfg["motion"]
        a = time_scale([arm_from, arm_to], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = ramp(hand_from, hand_to, mo["hand_max_vel"], self.dt)
        n = max(len(a), len(h))
        a = np.vstack([a, np.repeat(np.asarray(arm_to, float)[None], n - len(a), 0)]) if len(a) < n else a
        h = np.vstack([h, np.repeat(np.asarray(hand_to, float)[None], n - len(h), 0)]) if len(h) < n else h
        return a, h
