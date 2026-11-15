"""Tests for the lid and hole finder.  python -m unittest aaltonomous.brain.test_lid -v

Real photos (no robot camera pose: shapes and box only), empty boxes from the robot's head camera (no lid may be
found), and drawn scenes: a box and lid at a known 3D pose, seen by the head camera, to check heights, positions and
angles in the robot frame.
"""
import glob
import os
import unittest

import cv2
import numpy as np

from aaltonomous.brain.calibrate import CAL, TABLE_Z, bins
from aaltonomous.brain.colors import load
from aaltonomous.brain.lid import find_lid

DATA = os.path.join(os.path.dirname(__file__), "testdata")
WALL = 0.075
# holes in the lid's own frame (m), measured sizes (triangle 5.5 cm tall = 6.35 cm side): like the real lid, triangle at the back (corner pointing +y), circle and square
HOLES = {"prism": (0.0, 0.05), "cylinder": (-0.05, -0.045), "cube": (0.05, -0.045)}


def _poly(shape, size, n=48):
    if shape == "cylinder":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return size / 2 * np.c_[np.cos(t), np.sin(t)]
    if shape == "cube":
        return size / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    t = np.radians([90, 210, 330])   # triangle with a corner towards +y
    return size / np.sqrt(3) * np.c_[np.cos(t), np.sin(t)]


def scene(top, yaw, box="center", lid_bgr=(40, 40, 40), shift=(0, 0), fill=None, seed=0):
    """Head-camera image (design pose) of a box turned by yaw with the lid's top face at height `top` above the
    table, moved by `shift` (m) inside the box. fill: shape whose hole has a yellow object in it.
    Returns the image and the true hole positions {shape: (x, y)}."""
    T, K = CAL.cams["head"].T_pelvis_optical, CAL.cams["head"].K
    rng = np.random.default_rng(seed)
    img = np.clip(np.array([150, 165, 195]) + rng.normal(0, 4, (480, 640, 3)), 0, 255).astype(np.uint8)
    c, R = bins()[box][:, :2].mean(0), np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def px(xy, z):   # robot-frame points (N, 2) at height z -> image points, 1/16 px (cv2 shift=4)
        p = (np.c_[xy, np.full(len(xy), z)] - T[:3, 3]) @ T[:3, :3]
        return np.int32((K[:2, :2] @ (p[:, :2] / p[:, 2:]).T + K[:2, 2:]).T * 16)

    square = lambda h: np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T + c
    fp = lambda pts, col, im=img: cv2.fillPoly(im, [pts], col, cv2.LINE_AA, 4)
    fp(cv2.convexHull(np.r_[px(square(0.11), TABLE_Z), px(square(0.11), TABLE_Z + WALL)]), (40, 100, 235))
    opening = px(square(0.1075), TABLE_Z + WALL)
    fp(opening, (120, 160, 190))
    layer, z = img.copy(), TABLE_Z + top
    centre = c + R @ shift
    fp(px(square(0.10) - c + centre, z), lid_bgr, layer)
    truth = {}
    for s, off in HOLES.items():
        truth[s] = centre + R @ off
        fp(px(_poly(s, {"prism": 0.0635, "cube": 0.0625, "cylinder": 0.06}[s]) @ R.T + truth[s], z), (95, 130, 160), layer)
        if s == fill:
            fp(px(_poly("cube", 0.045) @ R.T + truth[s], z), (30, 200, 230), layer)
    inside = cv2.fillPoly(np.zeros(img.shape[:2], np.uint8), [opening], 1, cv2.LINE_AA, 4)   # the walls hide the rest
    img[inside > 0] = layer[inside > 0]
    return img, truth


def turn(a, b, period):
    """Smallest difference between two angles that repeat every `period`."""
    d = (a - b) % period
    return min(d, period - d)


class RealPhotos(unittest.TestCase):
    def test_lid_photos(self):
        for f, box in (("lid_white.jpg", "center"), ("lid_black.jpg", "center"), ("lid_three_boxes.jpg", "right")):
            lid = find_lid(load(os.path.join(DATA, f)), T=False)
            self.assertEqual((lid.box, sorted(h.shape for h in lid.holes)), (box, ["cube", "cylinder", "prism"]), f)
            self.assertFalse(any(h.filled for h in lid.holes), f)

    def test_slanted_view(self):   # phone at about the robot's angle: the front wall hides the round and square holes
        for f in ["lid_slanted.jpg"] + [os.path.basename(f) for f in glob.glob(os.path.join(DATA, "lid_video_*.jpg"))]:
            lid = find_lid(load(os.path.join(DATA, f)), T=False)
            self.assertEqual((lid.box, [h.shape for h in lid.holes]), ("left", ["prism"]), f)

    def test_no_lid_in_open_boxes(self):
        files = glob.glob(os.path.join(DATA, "head_*.jpg")) + glob.glob(os.path.join(DATA, "drop2_*.jpg"))
        self.assertEqual([f for f in files if find_lid(load(f)) is not None], [])


class Positions(unittest.TestCase):
    def check(self, top, yaw, box="center", hidden=(), tol=0.005, **kw):
        """tol: 5 mm with the whole lid in view (its width gives the height); 12 mm when part of it is hidden (the
        holes' size gives it)."""
        img, truth = scene(top, yaw, box, **kw)
        truth = {s: p for s, p in truth.items() if s not in hidden}   # behind the box's front wall: not reported
        lid = find_lid(img)
        self.assertIsNotNone(lid)
        self.assertEqual(lid.box, box)
        self.assertLess(abs(lid.z - TABLE_Z - top), tol)
        self.assertEqual(sorted(h.shape for h in lid.holes), sorted(truth))
        for h in lid.holes:
            self.assertLess(np.hypot(h.x - truth[h.shape][0], h.y - truth[h.shape][1]), tol, h.shape)
            if h.shape == "cube":
                self.assertLess(np.degrees(turn(h.yaw, yaw, np.pi / 2)), 6)
            if h.shape == "prism":   # its corner points +y in the lid's frame: direction yaw + 90 deg
                self.assertLess(np.degrees(turn(h.yaw, yaw + np.pi / 2, 2 * np.pi / 3)), 6)
        return lid

    def test_lid_on_the_rim(self):
        self.check(WALL + 0.01, 0.0)

    def test_lid_low_in_a_turned_box(self):
        self.check(0.03, np.radians(25), hidden=["cylinder"], tol=0.012)

    def test_white_lid_turned_round(self):
        self.check(0.05, np.radians(200), tol=0.012, lid_bgr=(225, 230, 230))

    def test_lid_moved_inside_the_box(self):
        self.check(0.02, np.radians(-15), hidden=["cylinder"], tol=0.012, shift=(0.004, -0.004))

    def test_hole_with_an_object(self):
        lid = self.check(WALL + 0.01, np.radians(10), fill="cube")
        self.assertEqual([h.shape for h in lid.holes if h.filled], ["cube"])

    def test_lid_cut_by_the_image_border(self):   # the left box is half outside the image: the holes give the height
        img, truth = scene(0.05, 0.0, "left")
        lid = find_lid(img)
        self.assertEqual(lid.box, "left")
        for h in lid.holes:
            self.assertLess(np.hypot(h.x - truth[h.shape][0], h.y - truth[h.shape][1]), 0.015, h.shape)


if __name__ == "__main__":
    unittest.main()
