"""3D position of each detected object, in the robot frame (pelvis: x forward, y left, z up; metres).

    p = locate(obj, depth_mm)        # with the head camera's depth image (robot, sim): exact
    p = locate(obj)                  # without depth (e.g. the dataset videos): from the known table and object size
    locate(obj, depth_mm, T=pose)    # pose: calibrate.fit_camera's result; default = the organisers' design pose

With depth: every pixel of the object becomes a 3D point; the top face is the highest points, and its centre is the
object's position (z = top of the object). Without depth: the line from the camera through the object's centre in the
image meets the height of the object's middle (table + 2.25 cm), because every object is 4.5 cm tall.
width_m(obj, p) says how wide the object looks at that distance: about 4.5 to 6.5 cm if the camera pose is right.
"""
import numpy as np

from aaltonomous.brain.calibrate import CAL, TABLE_Z

SIZE = 0.045   # every object: 4.5 cm wide and tall (organisers' scene file)


def _T(T):
    return CAL.cams["head"].T_pelvis_optical if T is None else T


def locate(obj, depth_mm=None, T=None):
    T, K = _T(T), CAL.cams["head"].K
    if depth_mm is not None:
        y0, x0 = obj.y + obj.h // 4, obj.x + obj.w // 4   # middle half of the box: the object, not its neighbours
        d = depth_mm[y0:y0 + obj.h // 2, x0:x0 + obj.w // 2].astype(float)
        v, u = np.nonzero(d > 0)                           # 0 = no depth there
        if len(u) >= 10:
            z = d[v, u] / 1000
            p = np.c_[(u + x0 - K[0, 2]) / K[0, 0] * z, (v + y0 - K[1, 2]) / K[1, 1] * z, z] @ T[:3, :3].T + T[:3, 3]
            top = p[p[:, 2] >= np.percentile(p[:, 2], 90) - 0.01]   # highest points (within 1 cm): the top face
            return np.median(top, axis=0)
    ray = T[:3, :3] @ [(obj.cx - K[0, 2]) / K[0, 0], (obj.cy - K[1, 2]) / K[1, 1], 1]
    s = (TABLE_Z + SIZE / 2 - T[2, 3]) / ray[2]
    return T[:3, 3] + s * ray + [0, 0, SIZE / 2]   # report the top of the object, as with depth


def width_m(obj, p, T=None):
    """How wide the object looks, in metres, at position p: a check on the camera pose."""
    return obj.w * np.linalg.norm(p - _T(T)[:3, 3]) / CAL.cams["head"].K[0, 0]
