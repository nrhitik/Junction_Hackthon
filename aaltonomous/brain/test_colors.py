"""Tests for colour detection.  python -m unittest aaltonomous.brain.test_colors

testdata/head_*.jpg: 35 frames of the robot's head camera (dataset zupermiez/g1_pick_cube, at least one from each
of its 20 episodes), picked by eye so that
all six objects (2 blue, 2 orange, 2 yellow) are visible; the robot arm and the bins are in view too.
"""
import collections
import glob
import os
import unittest

import cv2
import numpy as np

from aaltonomous.brain.colors import BGR, find_objects

FRAMES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "testdata", "head_*.jpg")))


def scene(*colors):
    """Synthetic 640x480 grey table with one 70 px square per colour (pure palette colours)."""
    img = np.full((480, 640, 3), 140, np.uint8)
    for i, c in enumerate(colors):
        cv2.rectangle(img, (40 + 100 * i, 200), (110 + 100 * i, 270), BGR[c], -1)
    return img


class Colors(unittest.TestCase):
    def test_head_camera_frames(self):
        self.assertEqual(len(FRAMES), 35)
        wrong = {os.path.basename(f): dict(c) for f in FRAMES
                 if (c := collections.Counter(o.color for o in find_objects(cv2.imread(f))))
                 != {"blue": 2, "orange": 2, "yellow": 2}}
        self.assertFalse(wrong)

    def test_phone_photo(self):   # other camera, orange-tinted table, 9 objects incl. prisms, some lying down
        img = cv2.imread(os.path.join(os.path.dirname(__file__), "testdata", "phone_table.jpg"))
        self.assertEqual(collections.Counter(o.color for o in find_objects(img)), {"blue": 3, "orange": 3, "yellow": 3})

    def test_objects_are_inside_their_boxes(self):
        for o in find_objects(cv2.imread(FRAMES[0])):
            self.assertTrue(o.x <= o.cx <= o.x + o.w and o.y <= o.cy <= o.y + o.h)

    def test_every_palette_colour(self):
        hues = ["red", "orange", "yellow", "green", "blue", "pink"]
        self.assertEqual([o.color for o in find_objects(scene(*hues))], hues)
        self.assertEqual([o.color for o in find_objects(scene("purple", "cyan"), ignore=())], ["purple", "cyan"])
        self.assertEqual([o.color for o in find_objects(scene("black", "white"), achromatic=True)], ["black", "white"])

    def test_cyan_and_black_white_are_off_by_default(self):
        self.assertEqual(find_objects(scene("cyan", "black", "white")), [])

    def test_empty_table(self):
        self.assertEqual(find_objects(scene()), [])


if __name__ == "__main__":
    unittest.main()
