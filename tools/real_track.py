"""Real robot: actual vs commanded grasp point at each grasp (hand closing), from the runner logs (arm_q vs arm_q_cmd)."""
import glob
import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.kinematics import ArmKinematics

pol = ClassicalPolicy(log=lambda m: None)
kin = pol.kin
rows = []
import os
for f in sorted(glob.glob("/data/real_v2/*.npz")):
    z = np.load(f)
    t, q, qc, h, hc = z["t"], z["arm_q"], z["arm_q_cmd"], z["hand_q"], z["hand_q_cmd"]
    for side, sl, hs in (("left", slice(0, 7), slice(0, 7)), ("right", slice(7, 14), slice(7, 14))):
        # grasp moment: the commanded hand starts closing (index/middle joints move > 0.3 rad toward closed)
        import os
        if os.environ.get("AT") == "release":   # release: thumb goes back (thumb1 |cmd| -> 0.5) while fingers straighten
            closing = (np.abs(hc[:, hs][:, 1]) > 0.45) & (np.abs(hc[:, hs][:, 3]) < 0.1)
        else:
            closing = np.abs(hc[:, hs][:, 3]) > 0.8
        starts = np.nonzero(closing[1:] & ~closing[:-1])[0] + 1
        for i in starts:
            j = i if os.environ.get("AT") == "release" else min(i + 8, len(t) - 1)
            pa = kin.grasp_pose(side, q[j])[0]
            pc = kin.grasp_pose(side, qc[j])[0]
            jerr = qc[j, sl] - q[j, sl]
            rows.append((f.split("/")[-1][9:15], side, t[j] - t[0], pc, pa, jerr))
for run, side, tt, pc, pa, je in rows:
    d = 1000 * (pa - pc)
    print(f"{run} {side:5s} t={tt:5.1f}  cmd ({pc[0]:.3f},{pc[1]:+.3f},{pc[2]:.3f})  actual-cmd = ({d[0]:+5.1f},{d[1]:+5.1f},{d[2]:+5.1f}) mm  "
          f"joint err (cmd-act) {np.round(je, 3)}")
D = np.array([1000 * (r[4] - r[3]) for r in rows])
for side in ("left", "right"):
    m = np.array([r[1] == side for r in rows])
    if m.any():
        print(f"MEAN {side}: actual-cmd = {np.round(D[m].mean(0), 1)} mm  (std {np.round(D[m].std(0), 1)}) n={m.sum()}")
