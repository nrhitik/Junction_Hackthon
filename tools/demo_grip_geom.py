"""Fingertip geometry (left palm frame, mm) of the demo grips vs ours: pinch axis, gap, grasp point."""
import numpy as np
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics

kin = ArmKinematics(load_config(None, {}))
G = {  # LEFT order: thumb0,1,2, middle0,1, index0,1
    "mid (ours)":            [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64],
    "preshape (ours)":       [0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4],
    "cube demo cmd":         [-0.04, 0.92, 0.16, -0.47, -1.15, -0.52, -1.14],
    "cube demo contact":     [-0.24, 0.86, 0.10, -0.42, -1.08, -0.49, -0.99],
    "cube demo open":        [-0.74, 0.92, 0.10, -0.29, 0.0, -0.34, 0.0],
    "prism demo cmd":        [-0.35, 0.92, 0.18, -0.32, -0.04, -0.56, -1.35],
    "prism demo contact":    [-0.23, 0.89, 0.10, -0.34, -0.05, -0.57, -1.16],
}
for k, h in G.items():
    t = kin.tips_in_palm("left", np.r_[np.asarray(h, float), np.zeros(7)]) * 1000  # rows: index, middle, thumb
    idx, mid, th = t
    fm = 0.5 * (idx + mid)
    ax = fm - th
    print(f"{k:20s} idx={idx.round(0)} mid={mid.round(0)} thumb={th.round(0)}")
    print(f"{'':20s} thumb->fingers {np.linalg.norm(ax):.0f}mm axis={np.round(ax / np.linalg.norm(ax), 2)}  "
          f"thumb->index {np.linalg.norm(idx - th):.0f}mm axis={np.round((idx - th) / np.linalg.norm(idx - th), 2)}  "
          f"grasp pt(thumb/fingers mid)={(0.5 * (fm + th)).round(0)} (thumb/index mid)={(0.5 * (idx + th)).round(0)}")
