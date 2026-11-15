"""Hull vs exact distance, shoulder_yaw <-> torso, over the original planner's plans on the grid."""
import os, numpy as np, pinocchio as pin
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.interface.safety import DEFAULT_URDF
pol = ClassicalPolicy(overrides={"perception.shape.enabled": False, "grasp.shape_grasp.enabled": False,
                                 "team_safety.enabled": False}, log=lambda m: None)
kin = pol.kin
d = os.path.dirname(str(DEFAULT_URDF))
def model(hull):
    gm = pin.buildGeomFromUrdf(kin.model, str(DEFAULT_URDF), pin.GeometryType.COLLISION, package_dirs=[d, os.path.dirname(d)])
    if hull:
        for g in gm.geometryObjects:
            if hasattr(g.geometry, "buildConvexHull"):
                g.geometry.buildConvexHull(True, "Qt"); g.geometry = g.geometry.convex
    ix = {g.name: i for i, g in enumerate(gm.geometryObjects)}
    gm.removeAllCollisionPairs()
    for s in ("left", "right"):
        gm.addCollisionPair(pin.CollisionPair(ix[f"{s}_shoulder_yaw_link_0"], ix["torso_link_0"]))
    return gm, pin.GeometryData(gm)
EX, HU = model(False), model(True)
sc, pl = pol.scene, pol.planner
L = sc.cfg["layout"]
pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in np.linspace(*L["region_x"], 5) for y in np.linspace(*L["region_y"], 3)]))
pairs = []
for p in pts:
    for color, b in pol.mapping.items():
        plan = pl.plan(Detection(color, np.r_[p[:2], sc.counter_z + pl.height], 500, 500, "depth"), pol.home.copy(), pol.post["open"].copy())
        if not plan.ok: continue
        for k in range(0, len(plan.arm), 5):
            q = kin.q0.copy(); q[kin.arm_iq] = plan.arm[k]; q[kin.hand_iq] = plan.hand[k]
            pin.framesForwardKinematics(kin.model, kin.data, q)
            r = []
            for gm, gd in (EX, HU):
                pin.updateGeometryPlacements(kin.model, kin.data, gm, gd, q)
                pin.computeDistances(gm, gd)
                r.append(min(x.min_distance for x in gd.distanceResults))
            pairs.append(r)
P = np.array(pairs) * 100
print(f"{len(P)} configs. home hull/exact:", end=" ")
q = kin.q0.copy(); q[kin.arm_iq] = pol.home
pin.framesForwardKinematics(kin.model, kin.data, q)
for gm, gd in (EX, HU):
    pin.updateGeometryPlacements(kin.model, kin.data, gm, gd, q); pin.computeDistances(gm, gd)
    print(f"{100 * min(x.min_distance for x in gd.distanceResults):.2f} cm", end=" ")
print()
for lo, hi in ((-10, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 100)):
    m = (P[:, 1] >= lo) & (P[:, 1] < hi)
    if m.any():
        print(f"hull {lo:5.1f}-{hi:5.1f} cm: n={m.sum():4d}  exact min {P[m, 0].min():5.2f}  median {np.median(P[m, 0]):5.2f}  "
              f"exact==0 (contact): {(P[m, 0] <= 1e-6).sum()}")
