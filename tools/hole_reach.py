"""IK reach at each lid hole: above_bin (carry height) and drop, with and without the hole orientation hint."""
import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.kinematics import ArmKinematics

pol = ClassicalPolicy(task="insert_shapes", log=print) if False else ClassicalPolicy(log=print)
pol.task = "insert_shapes"
holes = pol._lid_holes()
pl, kin, g, sc = pol.planner, pol.kin, pol.cfg["grasp"], pol.scene
zc, H = sc.counter_z, pl.height
wall = float(sc.cfg["bins"]["wall_height"])
z_carry = zc + max(g["carry_height"], wall + H + 0.03)
print("lid bin", sc.lid_bin, "z_carry", round(z_carry, 3), "wall", wall, "lid z", {k: round(v["z"], 3) for k, v in holes.items()})
for shp, h in holes.items():
    for side in ("left", "right"):
        for zname, z in (("above", z_carry), ("mid", z_carry - 0.05), ("drop", h["z"] + H / 2 + h["dz"])):
            p = np.r_[h["xy"], z]
            res = []
            kin.hint_signed = False
            for dn, d in [("free", None)] + [(f"d{i}", v) for i, v in enumerate(h["dirs"] or [])]:
                if d is not None:
                    kin.hint = np.r_[d, 0.0]
                _, ok, err, tilt = kin.solve(side, p, pol.home, top_down=True, extra_seeds=[pol.home[ArmKinematics.sl(side)]])
                res.append(f"{dn}:{'ok' if ok else 'X'}({1000*err:.0f}mm,{np.degrees(tilt):.0f}d)")
            print(f"{shp:8s} {side:5s} {zname:5s} p=({p[0]:.3f},{p[1]:+.3f},{p[2]:.3f}) " + " ".join(res))
