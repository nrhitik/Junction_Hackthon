"""Rung 3: find the lid with the shaped holes in the head camera image, and where each hole is in the robot frame.

    lid = find_lid(bgr, T)      # T: camera pose (calibrate.fit_camera; None = design pose; False = not the robot's
                                #    camera, e.g. a phone photo: image positions only). None: no lid in view.
    lid.box                     # 'left' / 'center' / 'right'
    lid.z                       # height of the lid's top face (robot frame, m)
    lid.hole("cube")            # Hole(shape, u, v, x, y, yaw, size, filled), or None if that hole is not fully in view
    lid.pairs                   # {('left', 'cube'): Hole, ...}: each hole named (box, shape)
    python -m aaltonomous.brain.lid image.jpg [out.jpg]

How it works, with no depth camera and no fixed layout (the box and lid can be turned, the lid can lie anywhere
between the box floor and its rim):
- Boxes: strongly orange regions (the table's orange tint is weaker). Inside each box the pixels are split into two
  colour groups; a group is the lid if it forms one solid plate enclosing hole-shaped patches (box floor seen through
  the holes). The lid's own colour does not matter (white, black, ...).
- Height: the lid is 20 cm wide. Traced back onto the table plane its outline looks wider the higher it lies (closer
  to the camera), so its width gives its height. If the lid is cut by the image border, the holes' 6 cm width is used.
- Hole shape: the outline is traced back onto the lid plane (taking out the camera's slant); a circle has every edge
  point equally far from its centre, a square fills its tightest rectangle (~0.95), a triangle does not (~0.65).
- yaw (rad, about the robot's z axis, from x towards y): square = direction of one edge, 0-90 deg; triangle =
  direction from its centre to a corner, 0-120 deg; circle = 0. Turn the object to the same yaw (modulo 90 / 120 deg).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from aaltonomous.brain.calibrate import CAL, TABLE_Z, bins
from aaltonomous.brain.colors import color_masks

LID, HOLE = 0.20, 0.06   # lid 20 x 20 x 1 cm (measured); square side / circle diameter of the holes (from photos)
TYPICAL = {"cylinder": (1.05, 0.8), "cube": (1.37, 0.97), "prism": (1.7, 0.63)}   # (roundness, fill)
W_L = 0.25               # weight of lightness against colour when splitting a box's inside into two groups


@dataclass
class Hole:
    shape: str
    u: float                  # image position (px)
    v: float
    x: float = None           # robot frame (m); None without a camera pose
    y: float = None
    yaw: float = None
    size: float = None        # square side / circle diameter / triangle side (m): ~0.06
    filled: bool = False      # something coloured (an object) is in the hole


@dataclass
class Lid:
    box: str
    z: float
    holes: list
    outline: np.ndarray       # image outline of the visible lid

    def hole(self, shape):
        return next((h for h in self.holes if h.shape == shape), None)

    @property
    def pairs(self):
        """{(box, shape): Hole}, e.g. ('left', 'cube'): pairs with an object's (colour, shape) through the plan's bin."""
        return {(self.box, h.shape): h for h in self.holes}


def boxes(bgr):
    """[(tape mask, [convex outline of each box])], from two ways of finding the boxes' orange tape:
    - the reddest thing in view: the redness (Lab a) that best splits the reddish pixels in two (Otsu) separates it from
      the table and the cardboard, whatever the camera's colours;
    - strongly saturated orange: also finds tape washed out by bright light, but can take in a strongly lit table.
    The lid search tries both; the one finding a lid with more holes wins."""
    a, c = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[..., 1], color_masks(bgr)
    out = []
    for m in (a > cv2.threshold(a[a > 138].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0],
              ((c["orange"] | c["red"]) > 0) & (cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1] > 130)):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        out.append((m, [cv2.convexHull(c) for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if cv2.contourArea(c) > 0.02 * m.size]))
    return out


def shape_of(pts):
    """'cube' / 'cylinder' / 'prism' from an outline (N, 2), any slant. The convex outline (holes are convex; this
    drops dents from shadows) is stretched back to round, then compared with each shape's (roundness, fill): roundness =
    farthest / nearest edge point from the centre, fill = area / tightest rectangle; typical values from the photos."""
    h = cv2.convexHull(np.float32(pts.reshape(-1, 2))).reshape(-1, 2).astype(float)
    h = (h - h.mean(0)) / h.std()
    m = cv2.moments(np.float32(h))   # centre and spread of the area (not of the edge points, which bunch up at corners)
    w, v = np.linalg.eigh(np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"])
    p = np.vstack([np.linspace(a, b, 2 + int(60 * np.hypot(*(b - a))), endpoint=False) for a, b in zip(h, np.roll(h, -1, 0))])
    q = (p - [m["m10"] / m["m00"], m["m01"] / m["m00"]]) @ v / np.sqrt(w)
    r, fill = np.hypot(*q.T), cv2.contourArea(np.float32(q)) / np.prod(cv2.minAreaRect(np.float32(q))[1])
    k = r.max() / r.min()
    return min(TYPICAL, key=lambda s: ((k - TYPICAL[s][0]) / 0.3) ** 2 + ((fill - TYPICAL[s][1]) / 0.15) ** 2)


def to_plane(uv, z, T):
    """Image points (N, 2) -> (N, 2) robot-frame x, y where their camera rays meet the horizontal plane at height z."""
    K = CAL.cams["head"].K
    d = np.c_[(uv - K[:2, 2]) / np.diag(K)[:2], np.ones(len(uv))] @ T[:3, :3].T
    return T[:2, 3] + d[:, :2] * ((z - T[2, 3]) / d[:, 2:])


def _plates(bgr):
    """Yield (box index, lid outline, [hole outlines], [hole filled?], box outlines) for every lid-like plate in a box.
    The tape rims cut the inside of each box outline into pieces (each box's opening, the table in front); each piece
    is split into three colour groups (e.g. lid, lit cardboard, shaded walls) and each group, and each pair of groups
    (a lid can be lit on one side and shaded on the other), is tried as the lid.
    Filled: with 3 holes in view, a hole whose colour is far from the others' (they show the box floor) has an object
    in it; with fewer holes there is nothing to compare with and none is marked."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32) * [W_L, 1, 1]
    k3, blank = np.ones((3, 3), np.uint8), np.zeros(bgr.shape[:2], np.uint8)
    solid = lambda c, a: cv2.contourArea(c) > a * cv2.contourArea(cv2.convexHull(c))
    fill = lambda c: cv2.drawContours(blank.copy(), [c], -1, 1, -1)
    pieces = [(i, outlines, np.uint8(piece == k)) for orange, outlines in boxes(bgr) for i, hull in enumerate(outlines)
              for n, piece, st, _ in [cv2.connectedComponentsWithStats(cv2.fillConvexPoly(blank.copy(), hull, 1) &
                                                                       (1 - cv2.dilate(orange, np.ones((5, 5), np.uint8))))]
              for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 0.01 * blank.size]
    for i, outlines, inside in pieces:
        px = lab[inside > 0]
        _, _, ctr = cv2.kmeans(np.float32(px[::4]), 3, None, (3, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        group = blank.copy()
        group[inside > 0] = np.linalg.norm(px[:, None] - ctr[None], axis=2).argmin(1) + 1
        for j in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3)):
            m = cv2.morphologyEx(np.uint8(np.isin(group, j)), cv2.MORPH_OPEN, k3)
            for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                a = cv2.contourArea(c)   # (front holes cut by the box wall bite into the outline: 0.75, not convex)
                if a < 0.08 * inside.sum() or not solid(c, 0.75):
                    continue
                gaps = cv2.morphologyEx(fill(c) & (1 - m), cv2.MORPH_OPEN, k3)
                holes = [h for h in cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                         if max(30, 0.02 * a) < cv2.contourArea(h) < 0.25 * a and solid(h, 0.85)]
                col = [np.median(lab[fill(h) > 0], axis=0) for h in holes]
                full = [len(holes) > 2 and np.linalg.norm(x - np.median(col, axis=0)) > 30 for x in col]
                if holes:
                    yield i, c, holes, full, outlines


def _measure(bgr, plate, T):
    """Lid from one plate: heights and positions with camera pose T (see the module notes), or None if no hole has a
    believable size (3.5-9 cm)."""
    i, c, holes, full, outlines = plate
    mid = lambda p: p.reshape(-1, 2).mean(0)
    if T is False:   # no camera pose: image positions; boxes named by their order (or by thirds of one wide box mask)
        order, x = sorted(range(len(outlines)), key=lambda j: mid(outlines[j])[0]), outlines[i][..., 0]
        third = int(np.clip(3 * (mid(c)[0] - x.min()) / np.ptp(x), 0, 2))
        box = ("left", "center", "right")[order.index(i) if len(outlines) == 3 else third if len(outlines) == 1 else 0]
        return Lid(box if len(outlines) in (1, 3) else None, None,
                   [Hole(shape_of(h), *mid(h), filled=f) for h, f in zip(holes, full)], c)
    H, W = bgr.shape[:2]
    flat = lambda pts, z=TABLE_Z: cv2.convexHull(np.float32(to_plane(pts.reshape(-1, 2).astype(float), z, T))).reshape(-1, 2)
    rect = cv2.minAreaRect(flat(c))
    if cv2.contourArea(flat(c)) > 0.95 * np.prod(rect[1]) and min(rect[1]) > 0.93 * max(rect[1]) and not np.any(
            (c[..., 0] <= 1) | (c[..., 1] <= 1) | (c[..., 0] >= W - 2) | (c[..., 1] >= H - 2)):
        scale = LID / max(rect[1])   # the whole lid is in view (a square on the table): its 20 cm width
    else:   # cut by a box wall, the hand or the image border: the round and square holes' 6 cm
        # (minus 1.5 px: blurred edge pixels count as hole, which makes holes look slightly wider)
        px = lambda h: np.linalg.norm(np.subtract(*to_plane(mid(h) + np.array([[0, 0], [1, 0]]), TABLE_Z, T)))
        sizes = [np.sqrt(cv2.contourArea(flat(h))) / (1 if shape_of(flat(h)) == "cube" else np.sqrt(np.pi) / 2)
                 - 1.5 * px(h) for h in holes if shape_of(flat(h)) != "prism"]
        if not sizes:
            return None
        scale = HOLE / np.median(sizes)
    z = T[2, 3] - (T[2, 3] - TABLE_Z) * scale   # the height where the lid (or hole) has its real width
    if not TABLE_Z < z < TABLE_Z + 0.15:
        return None
    out = []
    for h, f in zip(holes, full):
        p = flat(h, z)
        m = cv2.moments(np.float32(p * 1000))
        x, y, area = m["m10"] / m["m00"] / 1000, m["m01"] / m["m00"] / 1000, m["m00"] / 1e6
        shape = shape_of(p)
        if shape == "cube":
            yaw, size = np.radians(cv2.minAreaRect(np.float32(p * 1000))[2]) % (np.pi / 2), np.sqrt(area)
        elif shape == "prism":
            tip = p[np.argmax(np.hypot(p[:, 0] - x, p[:, 1] - y))]
            yaw, size = np.arctan2(tip[1] - y, tip[0] - x) % (2 * np.pi / 3), np.sqrt(4 * area / np.sqrt(3))
        else:
            yaw, size = 0.0, 2 * np.sqrt(area / np.pi)
        if 0.035 < size < 0.09:
            out.append(Hole(shape, *mid(h), x, y, float(yaw), float(size), f))
    centre = to_plane(mid(c)[None], z, T)[0]
    box = min(bins().items(), key=lambda b: np.linalg.norm(b[1][:, :2].mean(0) - centre))[0]
    return Lid(box, float(z), out, c) if out else None


def find_lid(bgr, T=None):
    """The lid with the most holes in view (see the module notes), or None."""
    T = CAL.cams["head"].T_pelvis_optical if T is None else T
    return max(filter(None, (_measure(bgr, p, T) for p in _plates(bgr))), key=lambda l: len(l.holes), default=None)


def draw(bgr, lid):
    out = bgr.copy()
    if lid:
        cv2.drawContours(out, [lid.outline], -1, (0, 255, 0), 2)
        for h in lid.holes:
            cv2.putText(out, f"{lid.box or '?'} {h.shape}" + (" full" if h.filled else ""), (int(h.u) - 35, int(h.v)), 0,
                        0.45, (255, 0, 255), 2)
    return out


if __name__ == "__main__":
    import sys
    from aaltonomous.brain.colors import load
    img = load(sys.argv[1])
    lid = find_lid(img)
    print("no lid seen" if lid is None else f"lid on the {lid.box} box, top at {lid.z - TABLE_Z:.3f} m above the table")
    for h in lid.holes if lid else []:
        print(f"  {h.shape:8s} x {h.x:.3f} y {h.y:.3f} yaw {np.degrees(h.yaw):5.1f} deg  {h.size * 100:.1f} cm"
              + ("  (something in it)" if h.filled else ""))
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], draw(img, lid))
