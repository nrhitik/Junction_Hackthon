"""Tests for shape detection.  python -m unittest aaltonomous.brain.test_shapes

testdata/shapes.json: 79 real objects labelled by eye (70 from the robot's head camera: cubes and cylinders; 9 from a
phone photo of the real table: cubes, cylinders and prisms, some lying down). The network never saw these photos.
"""
import unittest

import cv2
import numpy as np

from aaltonomous.brain import render, shapes
from aaltonomous.brain.colors import find_objects


class Shapes(unittest.TestCase):
    def test_real_photos(self):
        pred, truth = shapes.test()
        self.assertGreaterEqual(np.mean(pred == truth), 0.95)
        for k in range(3):   # every shape is recognised, and none is a dumping ground for the others
            self.assertGreaterEqual(np.mean(pred[truth == k] == k), 0.9)

    def test_one_answer_per_object(self):
        img = cv2.imread(f"{shapes.HERE}/testdata/phone_table.jpg")
        objs = find_objects(img)
        out = shapes.classify(img, objs)
        self.assertEqual(len(out), len(objs))
        self.assertTrue(all(s in render.SHAPES and 0 < p <= 1 for s, p in out))
        self.assertEqual(shapes.classify(img, []), [])

    def test_renderer(self):
        img, label = render.sample(np.random.default_rng(0), render.backgrounds())
        self.assertEqual((img.shape, img.dtype), ((64, 64, 3), np.uint8))
        self.assertIn(label, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
