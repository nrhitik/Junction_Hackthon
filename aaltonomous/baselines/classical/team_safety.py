"""Team safety layer: self-collision + no cross-body reach. Runs IN ADDITION to the challenge SafetyFilter.

The challenge filter (aaltonomous/interface/safety.py) checks joint limits, velocity, a wrist/palm workspace box and the
counter keep-out, but NOT self-collision: a plan could swing one arm through the other arm or the torso and still pass.
This layer checks, with the URDF collision meshes (convex hulls, hpp-fcl via pinocchio):
  - arm vs arm (shoulder_yaw .. fingertips, both sides),
  - arm vs body (torso, pelvis, head, logo),
  minimum distance >= `margin` (m); and
  - no cross-body reach: no left-arm point (elbow, wrist, palm, fingers) at pelvis y < -cross_margin, no right-arm
    point at y > +cross_margin.
Used by the planner (every candidate plan and feedback segment is rejected if any checked tick fails) and by the
policy as a last guard on every chunk before it is sent.
"""
from __future__ import annotations

import numpy as np

ARM_LINKS = ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
             "hand_palm_link", "hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")
BODY_LINKS = ("torso_link", "pelvis_contour_link", "head_link", "logo_link")
# points that must stay on their own side (link frames)
SIDE_POINTS = ("elbow_link", "wrist_roll_link", "wrist_yaw_link", "hand_palm_link", "hand_thumb_2_link",
               "hand_index_1_link", "hand_middle_1_link")


class TeamSafety:
    def __init__(self, cfg, kin):
        import pinocchio as pin
        from aaltonomous.interface.safety import DEFAULT_URDF
        import os
        self.pin, self.kin = pin, kin
        c = cfg.get("team_safety") or {}
        self.enabled = bool(c.get("enabled", True))
        self.margin = float(c.get("margin", 0.02))
        self.cross_margin = float(c.get("cross_margin", 0.06))
        self.check_every = int(c.get("check_every", 4))
        # per-link-pair margins, e.g. {"shoulder_yaw_link-torso_link": 0.01} (link names without the side prefix)
        self.pair_margins = {tuple(k.split("-")): float(v) for k, v in (c.get("pair_margins") or {}).items()}
        d = os.path.dirname(str(DEFAULT_URDF))
        self.gm = pin.buildGeomFromUrdf(kin.model, str(DEFAULT_URDF), pin.GeometryType.COLLISION,
                                        package_dirs=[d, os.path.dirname(d)])
        for g in self.gm.geometryObjects:  # meshes -> convex hulls: fast GJK distance, slightly conservative
            try:
                g.geometry.buildConvexHull(True, "Qt")
                g.geometry = g.geometry.convex
            except Exception:
                pass
        by_link = {}
        for i, g in enumerate(self.gm.geometryObjects):
            by_link.setdefault(kin.model.frames[g.parentFrame].name, []).append(i)
        L = {s: [i for n in ARM_LINKS for i in by_link.get(f"{s}_{n}", [])] for s in ("left", "right")}
        B = [i for n in BODY_LINKS for i in by_link.get(n, [])]
        pairs = [(a, b) for a in L["left"] for b in L["right"]]
        pairs += [(a, b) for s in ("left", "right") for a in L[s] for b in B]
        self.names = {i: self.gm.geometryObjects[i].name for i in range(self.gm.ngeoms)}
        self.gm.removeAllCollisionPairs()
        for a, b in pairs:
            self.gm.addCollisionPair(pin.CollisionPair(a, b))
        self.gd = pin.GeometryData(self.gm)
        self._set_margins()
        self.side_frames = {s: [kin.model.getFrameId(f"{s}_{n}") for n in SIDE_POINTS] for s in ("left", "right")}
        self.ignored = []

    def _set_margins(self):
        def link(i):
            n = self.names[i].rsplit("_", 1)[0]
            return n.split("_", 1)[1] if n.startswith(("left_", "right_")) else n
        self.pair_margin = []
        for p in self.gm.collisionPairs:
            a, b = link(p.first), link(p.second)
            self.pair_margin.append(self.pair_margins.get((a, b), self.pair_margins.get((b, a), self.margin)))
        self.pair_margin = np.array(self.pair_margin)

    def calibrate(self, poses):
        """Drop pairs already closer than the margin in known-safe poses (e.g. home: shoulder covers on the torso)."""
        bad = set()
        for arm_q, hand_q in poses:
            for k, d in self._distances(arm_q, hand_q):
                if d < self.pair_margin[k]:
                    bad.add(k)
        if bad:
            keep = [p for k, p in enumerate(self.gm.collisionPairs) if k not in bad]
            self.ignored = [(self.names[p.first], self.names[p.second]) for k, p in enumerate(self.gm.collisionPairs)
                            if k in bad]
            self.gm.removeAllCollisionPairs()
            for p in keep:
                self.gm.addCollisionPair(p)
            self.gd = self.pin.GeometryData(self.gm)
            self._set_margins()

    def _q(self, arm_q, hand_q):
        q = self.kin.q0.copy()
        q[self.kin.arm_iq] = arm_q
        if hand_q is not None:
            q[self.kin.hand_iq] = hand_q
        return q

    def _distances(self, arm_q, hand_q):
        pin = self.pin
        q = self._q(arm_q, hand_q)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        pin.updateGeometryPlacements(self.kin.model, self.kin.data, self.gm, self.gd, q)
        pin.computeDistances(self.gm, self.gd)
        return [(k, r.min_distance) for k, r in enumerate(self.gd.distanceResults)]

    def slack(self, arm_q, hand_q=None):
        """(worst slack m, reason): min over pairs of (distance - margin) and over side points of (cross_margin -
        crossing). >= 0 = safe."""
        ds = self._distances(np.asarray(arm_q, float), None if hand_q is None else np.asarray(hand_q, float))
        worst, why = np.inf, None
        if ds:
            k, d = min(ds, key=lambda x: x[1] - self.pair_margin[x[0]])
            worst = d - self.pair_margin[k]
            p = self.gm.collisionPairs[k]
            why = (f"self-collision {self.names[p.first]} <-> {self.names[p.second]} "
                   f"{100 * d:.1f} cm (< {100 * self.pair_margin[k]:.1f})")
        for s, sign in (("left", -1.0), ("right", 1.0)):
            for f in self.side_frames[s]:
                y = float(self.kin.data.oMf[f].translation[1])
                sl = self.cross_margin - sign * y  # left arm at y < -m, or right arm at y > +m -> negative
                if sl < worst:
                    worst, why = sl, f"cross-body: {s} {self.kin.model.frames[f].name} at y={y:+.3f}"
        return worst, why

    def check(self, arm_q, hand_q=None):
        """None if safe, else a short reason."""
        if not self.enabled:
            return None
        s, why = self.slack(arm_q, hand_q)
        return why if s < 0 else None

    def check_traj(self, arm, hand, every=None, start=None, tol=0.002):
        """First failing tick (index, reason) of a (T, 14) trajectory, checking every `every`-th tick and the last.
        start = (arm_q, hand_q) the motion begins from: if that pose itself is already inside a margin (e.g. the robot
        was put down at q = 0), ticks may stay that close but must not get WORSE (so moving away is allowed)."""
        if not self.enabled or arm is None or not len(arm):
            return None
        base = 0.0
        if start is not None:
            base = min(0.0, self.slack(*start)[0])
        n = len(arm)
        e = int(every or self.check_every)
        for k in list(range(0, n, e)) + [n - 1]:
            s, why = self.slack(arm[k], None if hand is None else hand[k])
            if s < base - (tol if base < 0 else 0.0):
                return k, why
        return None
