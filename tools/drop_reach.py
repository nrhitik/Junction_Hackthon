"""IK reach of the bin drop points with the bins moved +0.10 m (real setup), per inset / height above rim."""
import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.kinematics import ArmKinematics

pol = ClassicalPolicy(log=lambda m: None)
pol.task = "sort_fixed"
pol._apply_bin_shift(0.10)
pl, kin, g, sc = pol.planner, pol.kin, pol.cfg["grasp"], pol.scene
H, zc, wall = pl.height, sc.counter_z, float(sc.cfg["bins"]["wall_height"])
print("box fronts now at pelvis x", round(sc.mat_to_pelvis(np.r_[sc.cfg["bins"]["centers"]["center"][0],
      sc.cfg["bins"]["centers"]["center"][1] - 0.105, 0])[0], 3))
g["drop_inset"] = 0.035
for above in (0.04, 0.07, 0.10):
    row = []
    for b in ("left", "center"):
        xy = pl.drop_xy(b)
        z = zc + wall + H / 2 + above
        res = []
        for side in ("left", "right"):
            q, ok, err, tl = kin.solve(side, np.r_[xy, z], pol.home, top_down=False,
                                       extra_seeds=[pol.home[ArmKinematics.sl(side)]])
            res.append(f"{side[0]}:{'ok' if ok else 'X'}{1000 * err:.0f}/{np.degrees(tl):.0f}d")
        row.append(f"{b}({xy[0]:.3f},{xy[1]:+.3f}) {' '.join(res)}")
    print(f"ANY orientation above_rim {above:.2f} | " + " | ".join(row))
