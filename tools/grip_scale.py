import numpy as np
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec
cfg = load_config(None, {}); kin = ArmKinematics(cfg); sp = kin.spec
lo, hi = np.asarray(sp.hand_lower)[:7], np.asarray(sp.hand_upper)[:7]
print("left hand limits lo", lo.round(2), "hi", hi.round(2))
pre, cl = np.asarray(cfg["hand"]["preshape"]), np.asarray(cfg["hand"]["closed"])
for s in (1.0, 1.2, 1.3, 1.4, 1.5, 1.6):
    h = np.clip(pre + s * (cl - pre), lo + 0.06, hi - 0.06)
    t = kin.tips_in_palm("left", np.r_[h, np.zeros(7)]) * 1000
    f = t[:2].mean(0)
    print(f"scale {s}: {h.round(2).tolist()} gap={np.linalg.norm(f - t[2]):.0f}mm finger tip z(palm y)={t[0][1]:.0f} thumb={t[2].round(0)}")
