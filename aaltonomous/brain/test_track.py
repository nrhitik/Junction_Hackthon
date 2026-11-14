"""Tests for tracking and the background detection thread.  python -m unittest aaltonomous.brain.test_track"""
import os
import time
import unittest

import cv2

from aaltonomous.brain.colors import Obj
from aaltonomous.brain.track import Tracker, Watcher

FRAME = os.path.join(os.path.dirname(__file__), "testdata", "head_09360.jpg")


def obj(color, cx, cy, s=60):
    return Obj(color, cx - s // 2, cy - s // 2, s, s, s * s, cx, cy)


class Tracking(unittest.TestCase):
    def test_ids_stay_while_objects_move(self):
        tr = Tracker()
        for step in range(30):   # two orange objects side by side, one moving right 6 px per frame
            tracks = tr.update([obj("orange", 100 + 6 * step, 200), obj("orange", 400, 200)])
        self.assertEqual(sorted(t.id for t in tracks), [0, 1])
        self.assertEqual(tr.events, ["new 0 orange", "new 1 orange"])

    def test_flicker_is_never_reported(self):
        tr = Tracker(confirm=5)
        for _ in range(4):
            self.assertEqual(tr.update([obj("red", 50, 50)]), [])
        self.assertEqual(len(tr.update([obj("red", 50, 50)])), 1)   # fifth sighting: now it counts

    def test_gone_after_patience(self):
        tr = Tracker(patience=3, confirm=1)
        tr.update([obj("blue", 100, 100)], [("cube", 0.9)])
        for _ in range(3):
            self.assertEqual(tr.update([]), [])
            self.assertEqual(len(tr.tracks), 1)   # not seen, but still remembered
        tr.update([])
        self.assertEqual((tr.tracks, tr.events[-1]), ([], "gone 0 blue cube"))

    def test_shape_vote(self):
        tr = Tracker(confirm=1)
        for s, p in (("cube", 0.8), ("cube", 0.9), ("prism", 0.8), ("prism", 0.5), ("prism", 0.6)):
            t = tr.update([obj("yellow", 300, 300)], [(s, p)])[0]
        self.assertEqual(t.shape, "cube")   # the unsure "prism" answers don't vote

    def test_colours_are_not_mixed_up(self):
        tr = Tracker(confirm=1)
        tr.update([obj("blue", 100, 100)])
        tr.update([obj("yellow", 102, 100)])
        self.assertEqual(len(tr.tracks), 2)

    def test_background_thread(self):
        w, img = Watcher(), cv2.imread(FRAME)
        try:
            end = time.time() + 5
            while time.time() < end and len(w.objects) < 6:
                w.feed(img)
                time.sleep(0.03)   # like the robot's 30 Hz loop
            self.assertEqual(sorted(t.color for t in w.objects), ["blue"] * 2 + ["orange"] * 2 + ["yellow"] * 2)
            self.assertTrue(all(t.shape for t in w.objects))
        finally:
            w.stop()


if __name__ == "__main__":
    unittest.main()
