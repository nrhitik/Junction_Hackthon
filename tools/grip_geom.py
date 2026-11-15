"""Fingertip geometry (left palm frame, mm) of candidate Dex3 grips."""
import numpy as np
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
cfg = load_config(None, {})
kin = ArmKinematics(cfg)
G = {
 "base_preshape": cfg["hand"]["preshape"],
 "base_closed":   cfg["hand"]["closed"],
 "demo_open":     [-0.58, 0.92, 0.14, -0.42, 0.0, -0.29, 0.0],
 "demo_cmd":      [0.0, 0.92, 0.28, -0.49, -1.3, -0.37, -1.3],
 "demo_contact":  [-0.13, 0.83, 0.2, -0.42, -1.12, -0.35, -0.91],
}
for k, h in G.items():
    t = kin.tips_in_palm("left", np.r_[np.asarray(h, float), np.zeros(7)]) * 1000  # rows: index, middle, thumb
    f = t[:2].mean(0)
    print(f"{k:14s} tips(mm) idx={t[0].round(0)} mid={t[1].round(0)} thumb={t[2].round(0)} "
          f"| finger-thumb gap={np.linalg.norm(f - t[2]):.0f}mm  idx-mid={np.linalg.norm(t[0]-t[1]):.0f}mm "
          f"| grasp pt={(0.5*(f+t[2])).round(0)}")
print("config point_in_palm (mm):", np.round(np.asarray(cfg['grasp']['point_in_palm'])*1000))
