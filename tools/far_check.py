import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.kinematics import ArmKinematics
pol = ClassicalPolicy(log=lambda m: None)
pol.task = "sort_fixed"
pol._apply_bin_shift(0.108)
print([m for _, m in pol.events if "far boxes" in m])
pl, kin, sc = pol.planner, pol.kin, pol.scene
H, zc, wall = pl.height, sc.counter_z, float(sc.cfg["bins"]["wall_height"])
for name, v in pl.far_drop.items():
    zd = zc + wall + H / 2 + v["above"]
    for lab, z in (("drop", zd), ("above_bin(carry)", max(zc + 0.16, zd + 0.02))):
        r = []
        for sd in ("left", "right"):
            _, ok, err, tl = kin.solve(sd, np.r_[v["xy"], z], pol.home, top_down=True, extra_seeds=[pol.home[ArmKinematics.sl(sd)]])
            r.append(f"{sd}:{'ok' if ok else 'X'}{1000*err:.0f}")
        print(name, lab, np.round(v["xy"], 3), round(z, 3), r)
