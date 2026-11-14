"""Find every object on the table in a camera image and name its colour (OpenCV, no model).

    objs = find_objects(bgr)          # -> [Obj(color, x, y, w, h, area, cx, cy), ...]  pixels, image coordinates
    cv2.imwrite("out.jpg", annotate(bgr, objs))
    python -m aaltonomous.brain.colors image.jpg [out.jpg]

How: a pixel is "colourful" if it is saturated and bright enough; its hue gives the colour name (HUES). Connected
blobs of one colour whose size fits one object are objects. Blobs the size of two or three objects (same-coloured
objects that touch, seen as too big or not convex) are split; bigger ones (the bins) are dropped, and so are
same-coloured pieces in the bins' image rows (bin wall cut off by the robot's hand). Sizes are fractions of the image
area, tuned on the robot's head camera (640x480).

Black and white have no hue: they are only looked for with achromatic=True, and without depth the robot's own
black hands can look like black objects.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# colour name -> OpenCV hue range (0-179). Red wraps around.
HUES = [("red", 0, 8), ("orange", 9, 19), ("yellow", 20, 52), ("green", 53, 85), ("cyan", 86, 104), ("blue", 105, 128),
        ("purple", 129, 150), ("pink", 151, 169), ("red", 170, 179)]
PALETTE = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "black", "white")
BGR = {"red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255), "green": (0, 200, 0), "cyan": (255, 255, 0),
       "blue": (255, 0, 0), "purple": (200, 0, 150), "pink": (180, 105, 255), "black": (0, 0, 0),
       "white": (255, 255, 255)}


@dataclass
class Obj:
    color: str
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


def color_masks(bgr, s_min=75, v_min=70, achromatic=False):
    """{colour name: uint8 mask}. Colourful = saturated, or paler but very bright (sunlit yellow tops)."""
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    vivid = ((s >= s_min) & (v >= v_min)) | ((s >= 0.6 * s_min) & (v >= 175))
    out = {}
    for name, lo, hi in HUES:
        out[name] = out.get(name, 0) | (vivid & (h >= lo) & (h <= hi)).astype(np.uint8)
    if achromatic:
        out["black"] = (v < 45).astype(np.uint8)
        out["white"] = ((s < 30) & (v > 215)).astype(np.uint8)
    return out


def _split(mask, n):
    """Split one blob into n parts: k-means on its pixel coordinates."""
    pts = np.column_stack(np.nonzero(mask)).astype(np.float32)
    _, lab, _ = cv2.kmeans(pts, n, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1), 3,
                           cv2.KMEANS_PP_CENTERS)
    parts = []
    for i in range(n):
        m = np.zeros_like(mask)
        m[tuple(pts[lab[:, 0] == i].astype(int).T)] = 1
        parts.append(m)
    return parts


def find_objects(bgr, size=(0.004, 0.03), min_fill=0.45, ignore=("cyan",), **kw):
    """size = (min, max) area of ONE object as a fraction of the image; min_fill = blob area / bounding-box area.
    ignore: colours to skip. Default cyan: under the venue lights the robot's silver arm reads as cyan."""
    lo, hi = (f * bgr.shape[0] * bgr.shape[1] for f in size)
    objs = []
    for color, mask in color_masks(bgr, **kw).items():
        if color in ignore:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        big = st[1:][st[1:, cv2.CC_STAT_AREA] > 3 * hi]   # the bins: far too big to be objects
        rows = [(y, y + h) for _, y, _, h, _ in big]
        for i in range(1, n):
            area, cy = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] / 2
            if not lo <= area <= 3 * hi or any(y0 <= cy <= y1 for y0, y1 in rows):
                continue   # noise, a bin, or a piece of a bin wall cut off by the robot's hand
            blob = (lab == i).astype(np.uint8)
            hull = cv2.contourArea(cv2.convexHull(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]))
            if area < 0.65 * hull:
                continue   # hollow / L-shaped: a piece of a bin, not objects
            w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            # two touching objects: too big, or big and elongated, or big and not convex (the bigger, the stricter)
            merged = area > hi or (area > 0.55 * hi and (max(w, h) > 1.7 * min(w, h) or
                                                         area / hull < 0.85 + 0.1 * area / hi))
            for m in (_split(blob, max(2, int(np.ceil(area / hi)))) if merged else [blob]):
                x, y, w, h = cv2.boundingRect(m)
                a = int(m.sum())
                if a >= lo and a / (w * h) >= min_fill and 0.4 <= w / h <= 2.5:
                    ys, xs = np.nonzero(m)
                    objs.append(Obj(color, x, y, w, h, a, float(xs.mean()), float(ys.mean())))
    return sorted(objs, key=lambda o: (o.cx, o.cy))


def annotate(bgr, objs):
    out = bgr.copy()
    for o in objs:
        cv2.rectangle(out, (o.x, o.y), (o.x + o.w, o.y + o.h), BGR[o.color], 2)
        for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
            cv2.putText(out, o.color, (o.x, max(12, o.y - 4)), 0, 0.5, col, th)
    return out


if __name__ == "__main__":
    import sys
    img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), 1)
    found = find_objects(img)
    for o in found:
        print(o)
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], annotate(img, found))
