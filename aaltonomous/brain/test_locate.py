"""Tests for 3D positions and the box-based camera fit.  python -m unittest aaltonomous.brain.test_locate"""
import glob
import os
import unittest

import cv2
import numpy as np

from aaltonomous.brain import calibrate as C
from aaltonomous.brain.colors import Obj, find_objects
from aaltonomous.brain.locate import SIZE, locate, width_m

T0, K = C.CAL.cams["head"].T_pelvis_optical, C.CAL.cams["head"].K
FRAMES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "testdata", "head_*.jpg")))


def seen_at(p, half=25):
    """An Obj whose centre is where the camera sees the 3D point p."""
    q = (p - T0[:3, 3]) @ T0[:3, :3]
    u, v = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
    return Obj("orange", int(u) - half, int(v) - half, 2 * half, 2 * half, 4 * half * half, u, v)


class Locate(unittest.TestCase):
    def test_without_depth(self):   # an object whose middle is at (0.30, 0.10): found there, top reported
        p = locate(seen_at(np.array([0.30, 0.10, C.TABLE_Z + SIZE / 2])))
        np.testing.assert_allclose(p, [0.30, 0.10, C.TABLE_Z + SIZE], atol=0.002)

    def test_with_depth(self):      # a flat top face 4.5 cm above the table, in a made-up depth image
        top = np.array([0.28, -0.05, C.TABLE_Z + SIZE])
        o = seen_at(top)
        depth = np.zeros((480, 640), np.uint16)
        v, u = np.mgrid[o.y:o.y + o.h, o.x:o.x + o.w]
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1) @ T0[:3, :3].T
        depth[o.y:o.y + o.h, o.x:o.x + o.w] = (top[2] - T0[2, 3]) / ray[..., 2] * 1000   # distance along the view
        np.testing.assert_allclose(locate(o, depth), top, atol=0.005)

    def test_camera_fit_on_real_frames(self):
        for f in FRAMES[::9]:
            img = cv2.imread(f)
            T, wall, score = C.fit_camera(img)
            self.assertGreater(score, C.overlap(*C.box_mask(img), T0) + 0.3)   # boxes match far better than before
            self.assertTrue(0.04 <= wall <= 0.16)
            xs = [locate(o, T=T)[0] for o in find_objects(img)]   # objects lie 26-33 cm ahead (scene file)
            self.assertTrue(0.20 < np.median(xs) < 0.36, np.median(xs))
            ws = [width_m(o, locate(o, T=T), T) for o in find_objects(img)]
            self.assertTrue(0.04 < np.median(ws) < 0.075, np.median(ws))


if __name__ == "__main__":
    unittest.main()
