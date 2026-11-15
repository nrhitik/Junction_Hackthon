"""Offline: which arm plans each pick under arm_select=nearest vs bin (no sim). Objects on a grid over the layout region.

    aalto py /data/team/tools/arm_select_check.py
"""
import numpy as np
from collections import Counter

from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection

res = {}
for mode in ("bin", "nearest"):
    pol = ClassicalPolicy(overrides={"grasp.arm_select": mode, "perception.shape.enabled": False}, log=lambda m: None)
    sc, pl = pol.scene, pol.planner
    L = sc.cfg["layout"]
    pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in np.linspace(*L["region_x"], 5)
                                     for y in np.linspace(*L["region_y"], 3)]))
    hand = pol.post["open"]
    c, rows = Counter(), []
    for p in pts:
        for color, b in pol.mapping.items():
            det = Detection(color, np.r_[p[:2], sc.counter_z + pl.height], 500, 500, "depth")
            plan = pl.plan(det, pol.home.copy(), hand.copy())
            near = min(("left", "right"), key=lambda s: np.linalg.norm(pol.kin.grasp_pose(s, pol.home)[0][:2] - p[:2]))
            kind = ("FAIL" if not plan.ok else "relocate" if plan.target_bin == "counter" else
                    "near" if plan.side == near else "far")
            c[kind] += 1
            c["dur"] += plan.duration if plan.ok else 0
            rows.append((round(p[0], 2), round(p[1], 2), b, plan.side, kind, round(plan.duration, 1)))
    res[mode] = rows
    print(f"{mode:8s} {dict((k, v) for k, v in c.items() if k != 'dur')}  total plan time {c['dur']:.0f}s "
          f"over {len(rows)} picks")
diff = [(a, b) for a, b in zip(res["bin"], res["nearest"]) if a[3:5] != b[3:5]]
print(f"{len(diff)} picks change arm/kind:")
for a, b in diff:
    print(f"  obj ({a[0]:+.2f},{a[1]:+.2f}) -> {a[2]:6s}: bin-rule {a[3]}/{a[4]} {a[5]}s  ->  nearest {b[3]}/{b[4]} {b[5]}s")
