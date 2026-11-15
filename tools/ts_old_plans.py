"""Run the team safety layer over the ORIGINAL planner's plans on a grid of objects: what would it have rejected?"""
import numpy as np
from collections import Counter
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.team_safety import TeamSafety
pol = ClassicalPolicy(overrides={"perception.shape.enabled": False, "grasp.shape_grasp.enabled": False}, log=lambda m: None)
ts = TeamSafety(pol.cfg, pol.kin)
sc, pl = pol.scene, pol.planner
L = sc.cfg["layout"]
pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in np.linspace(*L["region_x"], 5) for y in np.linspace(*L["region_y"], 3)]))
c = Counter()
for p in pts:
    for color, b in pol.mapping.items():
        det = Detection(color, np.r_[p[:2], sc.counter_z + pl.height], 500, 500, "depth")
        plan = pl.plan(det, pol.home.copy(), pol.post["open"].copy())
        if not plan.ok:
            c["no plan"] += 1; continue
        # min distance over the plan
        mind = min(min(d for _, d in ts._distances(plan.arm[k], plan.hand[k])) for k in range(0, len(plan.arm), 4))
        bad = ts.check_traj(plan.arm, plan.hand, every=2)
        c["unsafe" if bad else "safe"] += 1
        if bad:
            print(f"obj ({p[0]:+.2f},{p[1]:+.2f}) -> {b:6s} {plan.side:5s} arm: tick {bad[0]}/{len(plan.arm)} {bad[1]}  (min dist {100*mind:.1f} cm)")
print(dict(c))
