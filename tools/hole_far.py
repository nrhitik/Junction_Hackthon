import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.kinematics import ArmKinematics
pol = ClassicalPolicy(log=lambda m: None)
pol.reset({"instruction": "Put each object through the hole of the same shape in the lid.", "task": "insert_shapes"})
H = pol.planner.height
for shift in (0.0, 0.108):
    pol._apply_bin_shift(shift)
    for shp, h in pol.planner.insert_holes.items():
        res = []
        for dz in (0.02, 0.04, 0.06, 0.08, 0.10):
            p = np.r_[h["xy"], h["z"] + H / 2 + dz]
            _, ok, err, _ = pol.kin.solve("left", p, pol.home, top_down=True, extra_seeds=[pol.home[ArmKinematics.sl("left")]])
            res.append(f"{100*dz:.0f}cm:{'ok' if ok else 'X'}{1000*err:.0f}")
        print(f"shift {100*shift:4.1f} cm  {shp:8s} hole ({h['xy'][0]:.3f},{h['xy'][1]:+.3f})  left arm: " + " ".join(res))
