"""Pixel -> point on the counter plane (pelvis frame) with the policy's calibration (least squares on project())."""
import sys
import numpy as np
from scipy.optimize import least_squares
from aaltonomous.baselines.classical.policy import ClassicalPolicy

pol = ClassicalPolicy(log=lambda m: None)
cal, cam, sc = pol.perception.cal, pol.perception.cam, pol.scene
zc = sc.counter_z

def at(u, v, z=zc):
    f = lambda xy: np.asarray(cal.project(cam, np.r_[xy, z])[:2]) - [u, v]
    return least_squares(f, [0.45, 0.0]).x

pts = {  # pixel (640x480 frame) of real features, read off the overlay image
    "left box   front-bottom, left end":   (12, 268),
    "left box   front-bottom, right end":  (128, 286),
    "center box front-bottom, left end":   (300, 228),
    "center box front-bottom, right end":  (510, 232),
    "right box  front-bottom, left end":   (520, 228),
    "center box rim front-left corner":    (305, 150),
    "yellow cube bottom-front":            (300, 322),
    "purple cube bottom-front":            (505, 365),
}
print(f"counter z (pelvis) = {zc:.3f}; model box front edge x = {sc.mat_to_pelvis(np.r_[sc.cfg['bins']['centers']['center'][0], sc.cfg['bins']['centers']['center'][1] - sc.cfg['bins']['outer_size'][1] / 2, 0])[0]:.3f}")
for k, (u, v) in pts.items():
    x, y = at(u, v)
    print(f"{k:38s} pixel ({u},{v}) -> pelvis x {x:.3f}  y {y:+.3f}")
