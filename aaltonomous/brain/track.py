"""Detection that keeps running: objects get a stable id from frame to frame, and the work happens in the background.

    w = Watcher()                    # starts a background thread
    w.feed(bgr)                      # from the robot's 30 Hz loop: hand over the newest image, returns at once
    w.objects                        # latest list of Track (id, color, shape, x, y, w, h, ...), never blocks
    w.events                         # ["new 3 orange cube", "gone 1 blue cylinder", ...] since the last read
    w.stop()

Tracking: each new detection is matched to the nearest track of the same colour (within half a box size + 40 px);
unmatched detections start new tracks, tracks unseen for `patience` updates are dropped. A track is only reported
after `confirm` sightings, so one-frame flickers (box edges while the arm moves) never show up. Box and position are
smoothed; the shape is the most frequent of the last confident answers, ignoring objects cut by the image border.
"""
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from aaltonomous.brain.colors import find_objects
from aaltonomous.brain.shapes import classify


@dataclass
class Track:
    id: int
    color: str
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    shapes: deque = field(default_factory=lambda: deque(maxlen=9))
    missed: int = 0
    seen: int = 1

    @property
    def shape(self):
        return Counter(self.shapes).most_common(1)[0][0] if self.shapes else None


class Tracker:
    def __init__(self, patience=10, smooth=0.5, confirm=5):
        self.tracks, self.next_id, self.events = [], 0, []
        self.patience, self.smooth, self.confirm = patience, smooth, confirm

    def update(self, objs, shapes=None):
        free = list(self.tracks)
        for o, s in zip(objs, shapes or [None] * len(objs)):
            near = [t for t in free if t.color == o.color and
                    np.hypot(t.cx - o.cx, t.cy - o.cy) < max(o.w, o.h) / 2 + 40]
            if near:
                t = min(near, key=lambda t: np.hypot(t.cx - o.cx, t.cy - o.cy))
                free.remove(t)
                a = self.smooth
                for k in ("x", "y", "w", "h", "cx", "cy"):
                    setattr(t, k, a * getattr(t, k) + (1 - a) * getattr(o, k))
                t.missed, t.seen = 0, t.seen + 1
            else:
                t = Track(self.next_id, o.color, o.x, o.y, o.w, o.h, o.cx, o.cy)
                self.next_id += 1
                self.tracks.append(t)
            if s and s[1] >= 0.7:   # only confident answers vote
                t.shapes.append(s[0])
            if t.seen == self.confirm:
                self.events.append(f"new {t.id} {t.color}")
        for t in free:
            t.missed += 1
        for t in [t for t in self.tracks if t.missed > self.patience]:
            self.tracks.remove(t)
            if t.seen >= self.confirm:
                self.events.append(f"gone {t.id} {t.color} {t.shape or ''}".strip())
        return [t for t in self.tracks if t.missed == 0 and t.seen >= self.confirm]


def detect(bgr):
    """Objects and their shapes; no shape for objects cut by the image border (a cut cube looks like a prism)."""
    objs = find_objects(bgr)
    H, W = bgr.shape[:2]
    return objs, [None if o.x <= 1 or o.y <= 1 or o.x + o.w >= W - 1 or o.y + o.h >= H - 1 else s
                  for o, s in zip(objs, classify(bgr, objs))]


class Watcher:
    """Runs detection + shapes + tracking on the newest fed image in a background thread."""

    def __init__(self, shapes=True, **kw):
        self.tracker, self.shapes, self._img, self._lock = Tracker(**kw), shapes, None, threading.Lock()
        self.objects, self.hz, self._run = [], 0.0, True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def feed(self, bgr):
        with self._lock:
            self._img = bgr

    @property
    def events(self):
        out, self.tracker.events = self.tracker.events, []
        return out

    def _loop(self):
        while self._run:
            with self._lock:
                img, self._img = self._img, None
            if img is None:
                time.sleep(0.005)
                continue
            t = time.perf_counter()
            objs, shapes = detect(img) if self.shapes else (find_objects(img), None)
            self.objects = self.tracker.update(objs, shapes)
            self.hz = 0.8 * self.hz + 0.2 / max(time.perf_counter() - t, 1e-3)

    def stop(self):
        self._run = False
        self._thread.join(1)
