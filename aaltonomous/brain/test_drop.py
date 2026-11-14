"""Tests for drop positions.  python -m unittest aaltonomous.brain.test_drop -v"""
import unittest

import numpy as np

from aaltonomous.brain.calibrate import TABLE_Z, bins
from aaltonomous.brain.drop import WALL, drop_point
from aaltonomous.brain.lid import find_lid
from aaltonomous.brain.test_lid import scene


class Drops(unittest.TestCase):
    def test_open_box_spots_stay_inside_and_apart(self):
        c = bins()["right"][:, :2].mean(0)
        spots = [drop_point("right", "cube", k=k) for k in range(5)]
        for d in spots:   # object (2.25 cm half width) inside the 21.5 cm box even if the box is turned 45 deg
            self.assertLess(np.hypot(d.x - c[0], d.y - c[1]) + 0.0225, 0.1075)
            self.assertGreater(d.z, TABLE_Z + WALL + 0.0225)
            self.assertIsNone(d.yaw)
        gaps = [np.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(spots) for b in spots[i + 1:]]
        self.assertGreaterEqual(min(gaps), 0.05 - 1e-9)
        self.assertEqual(drop_point("right", k=5), spots[0])   # then around again
        self.assertEqual(drop_point("right", centre=(0.5, -0.3))[:2], (0.5, -0.3))   # a measured box position

    def test_lid_box_uses_the_matching_hole(self):
        img, truth = scene(0.085, np.radians(20))
        lid = find_lid(img)
        for shape, (x, y) in truth.items():
            d = drop_point("center", shape, lid=lid)
            self.assertLess(np.hypot(d.x - x, d.y - y), 0.005)
            self.assertAlmostEqual(d.z, lid.z + 0.0425)
            self.assertEqual(d.yaw, lid.hole(shape).yaw)
        self.assertIsNotNone(drop_point("left", "cube", lid=lid).z)   # other boxes are open

    def test_hidden_hole(self):
        img, _ = scene(0.03, np.radians(25))   # the round hole is behind the box's front wall
        self.assertIsNone(drop_point("center", "cylinder", lid=find_lid(img)))


if __name__ == "__main__":
    unittest.main()
