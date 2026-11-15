"""Offline (no sim): plan every (grid object, bin) with the CURRENT config; outcome + which arm + duration."""
import sys, json, time
import numpy as np
from collections import Counter
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
pol = ClassicalPolicy(overrides=ov, log=lambda m: None)
sc, pl = pol.scene, pol.planner
L = sc.cfg["layout"]
pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in np.linspace(*L["region_x"], 5) for y in np.linspace(*L["region_y"], 3)]))
c, reasons = Counter(), Counter()
t0 = time.time()
for p in pts:
    for color, b in pol.mapping.items():
        det = Detection(color, np.r_[p[:2], sc.counter_z + pl.height], 500, 500, "depth")
        near = "left" if p[1] > 0 else "right"
        plan = pl.plan(det, pol.home.copy(), pol.post["open"].copy())
        kind = "SKIP" if not plan.ok else "relocate" if plan.target_bin == "counter" else ("near" if plan.side == near else "far")
        c[kind] += 1
        if not plan.ok:
            reasons[plan.reason[:110]] += 1
            print(f"  SKIP obj ({p[0]:+.2f},{p[1]:+.2f}) -> {b}: {plan.reason[:200]}")
print(dict(c), f"planning {time.time() - t0:.0f}s for {sum(c.values())} picks")
