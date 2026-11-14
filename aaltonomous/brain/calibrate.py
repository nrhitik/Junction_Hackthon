"""Find the head camera's real pose from the three boxes, which stand at known places on the counter.

    T, score = fit_camera(bgr)     # 4x4 pose of the camera in the robot frame (pelvis: x forward, y left, z up)
    python -m aaltonomous.brain.calibrate image.jpg [out.jpg]      # fit, print the change, draw the boxes

The organisers' pose (organisers/calibration.yaml) comes from the robot's design file. Starting from it, the angle and
position are adjusted until the outlines of the boxes, drawn from their known 3D positions (organisers/scene.yaml),
cover the orange boxes in the image as well as possible (overlap = intersection / union, 1.0 = perfect).
"""
import os

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from aaltonomous.brain.colors import color_masks, find_objects
from aaltonomous.organisers.calibration import Calibration

ORG = os.path.join(os.path.dirname(__file__), "..", "organisers")
CAL = Calibration.load(os.path.join(ORG, "calibration.yaml"))
SCENE = yaml.safe_load(open(os.path.join(ORG, "scene.yaml")))
TABLE_Z = SCENE["counter"]["mat_origin_in_pelvis"][2]   # counter surface height in the robot frame (m)


def bins(wall=None):
    """{name: 8 corners (robot frame, m)} of each box, from the scene file (mat coordinates -> robot frame).
    wall: wall height (m); the scene file's 6 cm is an unmeasured default."""
    b, (mx, my, _) = SCENE["bins"], SCENE["counter"]["mat_origin_in_pelvis"]
    hx, hy = np.asarray(b["outer_size"]) / 2
    out = {}
    for name, (u, v) in b["centers"].items():
        cx, cy = mx + v, my + u   # mat x runs along the robot's left, mat y away from it
        out[name] = np.array([[cx + sx * hy, cy + sy * hx, z] for z in (TABLE_Z, TABLE_Z + (wall or b["wall_height"]))
                              for sx in (-1, 1) for sy in (-1, 1)])
    return out


def box_mask(bgr):
    """(boxes, ignore): the boxes' orange/red pixels, and pixels not to judge: detected objects and the robot's dark
    hand, which can hide the boxes."""
    m = color_masks(bgr)
    m = ((m["red"] | m["orange"]) > 0).astype(np.uint8)
    ignore = (cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] < 70).astype(np.uint8)
    for o in find_objects(bgr):
        ignore[o.y:o.y + o.h, o.x:o.x + o.w] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(m & (1 - ignore))
    return np.isin(lab, 1 + np.flatnonzero(st[1:, cv2.CC_STAT_AREA] > 0.005 * m.size)).astype(np.uint8), ignore


def drawn(T, shape, wall=None):
    """The boxes' orange as the camera with pose T would see it: each box's outline minus its open top, through
    which the brown cardboard inside shows."""
    out = np.zeros(shape, np.uint8)
    K = CAL.cams["head"].K
    for c in bins(wall).values():
        p = (c - T[:3, 3]) @ T[:3, :3]
        uv = np.int32((K[:2, :2] @ (p[:, :2] / p[:, 2:]).T + K[:2, 2:]).T)
        cv2.fillConvexPoly(out, cv2.convexHull(uv), 1)
        cv2.fillConvexPoly(out, uv[[4, 5, 7, 6]], 0)   # top opening (corners at rim height)
    return out


def _pose(T0, x):
    """Change T0 by x = (3 rotation angles in rad, 3 position offsets in m)."""
    T = T0.copy()
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ T0[:3, :3]
    T[:3, 3] += x[3:]
    return T


def overlap(seen, ignore, T, wall=None):
    d = drawn(T, seen.shape, wall) & (1 - ignore)
    return np.sum(seen & d) / max(np.sum(seen | d), 1)


def fit_camera(bgr, T0=None, max_turn=0.35, max_move=0.15):
    """(pose, box wall height, overlap): best camera pose within max_turn rad (20 deg) and max_move m of T0.
    Starts from a coarse search over camera tilt (about its own x axis) and wall height, then refines all 7."""
    T0 = CAL.cams["head"].T_pelvis_optical if T0 is None else T0
    seen, ignore = box_mask(bgr)
    tilt = T0[:3, 0]   # the camera's own x axis (image right): turning about it tilts the view up / down
    cost = lambda x: 1 - overlap(seen, ignore, _pose(T0, np.r_[x[:3], x[3:6]]), x[6])
    grid = [np.r_[a * tilt, 0, 0, 0, w] for a in np.linspace(-max_turn, max_turn, 15) for w in (0.06, 0.09, 0.12)]
    x0 = min(grid, key=cost)
    bounds = [(-max_turn, max_turn)] * 3 + [(-max_move, max_move)] * 3 + [(0.04, 0.16)]
    best = minimize(cost, x0, method="Powell", bounds=bounds, options={"xtol": 1e-4, "ftol": 1e-5})
    return _pose(T0, best.x[:6]), float(best.x[6]), 1 - best.fun


def draw(bgr, T, wall=None):
    out = bgr.copy()
    cv2.drawContours(out, cv2.findContours(drawn(T, bgr.shape[:2], wall), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
                     -1, (0, 255, 0), 2)
    return out


def describe(T, T0=None):
    """How far T is from the design pose: angle (deg) and position (cm) changes, in words."""
    T0 = CAL.cams["head"].T_pelvis_optical if T0 is None else T0
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec()))
    down = lambda M: np.degrees(np.arcsin(-M[2, 2]))   # optical axis below horizontal
    dx, dy, dz = (T[:3, 3] - T0[:3, 3]) * 100
    return (f"turned {angle:.1f} deg, looks {down(T):.1f} deg down (design {down(T0):.1f}); "
            f"moved {dx:+.1f} cm forward, {dy:+.1f} cm left, {dz:+.1f} cm up")


if __name__ == "__main__":
    import sys
    from aaltonomous.brain.colors import load
    img = load(sys.argv[1])
    T, wall, score = fit_camera(img)
    print(f"box overlap {overlap(*box_mask(img), CAL.cams['head'].T_pelvis_optical):.2f} with the design pose, "
          f"{score:.2f} after fitting; walls {wall * 100:.0f} cm high; " + describe(T))
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], draw(img, T, wall))
