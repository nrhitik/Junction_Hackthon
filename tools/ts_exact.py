"""Exact-mesh (BVH) vs convex-hull distance on the worst ticks of a few original plans."""
import os, numpy as np, pinocchio as pin
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.team_safety import TeamSafety
from aaltonomous.interface.safety import DEFAULT_URDF
pol = ClassicalPolicy(overrides={"perception.shape.enabled": False, "grasp.shape_grasp.enabled": False}, log=lambda m: None)
ts = TeamSafety(pol.cfg, pol.kin)
d = os.path.dirname(str(DEFAULT_URDF))
gm = pin.buildGeomFromUrdf(pol.kin.model, str(DEFAULT_URDF), pin.GeometryType.COLLISION, package_dirs=[d, os.path.dirname(d)])
gm.removeAllCollisionPairs()
for p in ts.gm.collisionPairs:
    gm.addCollisionPair(pin.CollisionPair(p.first, p.second))
gd = pin.GeometryData(gm)
sc, pl = pol.scene, pol.planner
for (x, y, b) in [(0.26, 0.0, "left"), (0.30, 0.17, "right"), (0.33, 0.17, "right"), (0.26, 0.17, "left"), (0.30, 0.09, "left")]:
    p = sc.mat_to_pelvis(np.array([[0, 0, 0.0]]))[0]  # unused
    det = Detection("red", np.array([x, y, sc.counter_z + pl.height]), 500, 500, "depth")
    pol.mapping = {"red": b}; pl.mapping = pol.mapping
    plan = pl.plan(det, pol.home.copy(), pol.post["open"].copy())
    worst = (1e9, None, None)
    for k in range(0, len(plan.arm), 2):
        q = ts._q(plan.arm[k], plan.hand[k])
        pin.framesForwardKinematics(pol.kin.model, pol.kin.data, q)
        pin.updateGeometryPlacements(pol.kin.model, pol.kin.data, gm, gd, q)
        pin.computeDistances(gm, gd)
        for i, r in enumerate(gd.distanceResults):
            if r.min_distance < worst[0]:
                pr = gm.collisionPairs[i]
                worst = (r.min_distance, k, (gm.geometryObjects[pr.first].name, gm.geometryObjects[pr.second].name))
        cw = min(dd for _, dd in ts._distances(plan.arm[k], plan.hand[k]))
    ph = next(n for n, s, e in plan.phases if s <= worst[1] < e)
    print(f"obj ({x:+.2f},{y:+.2f})->{b:5s} {plan.side} arm: EXACT min {100*worst[0]:+.1f} cm at tick {worst[1]} ({ph}) {worst[2]}")
