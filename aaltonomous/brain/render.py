"""Synthetic training images of the three shapes, from the organisers' 3D models (no 3D library needed).

    img, label = sample(rng, backgrounds)   # 64x64 BGR crop around one rendered object; label 0 cube, 1 cylinder, 2 triangle
    python -m aaltonomous.brain.render out.jpg   # a sheet of examples

Each object rests on one of its faces (cylinders and prisms upright or lying down) with a random turn, colour and
light, and is seen by a camera from a random height, angle and distance. The objects are convex, so drawing the faces
that point at the camera is enough (no depth sorting). A shadow is cast onto a background patch cut from a real
photo of the table. The crop is placed around the object the way colors.py would box it, with some error.
"""
import glob
import os
import re

import cv2
import numpy as np

SHAPES = ("cube", "cylinder", "triangle")
HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "..", "organisers", "assets")
# resting faces: outward normals (object frame) that can point down
REST = {"cube": [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0)], "cylinder": [(0, 0, 1), (0, 0, -1), "side"],
        "triangle": [(0, 0, 1), (0, 0, -1), (0, -1, 0), (0.866, 0.5, 0), (-0.866, 0.5, 0)]}


def load_mesh(shape):
    """(n, 3, 3) triangles in metres, centred."""
    with open(os.path.join(ASSETS, f"object_{shape}.stl")) as f:
        v = np.array(re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", f.read()), float).reshape(-1, 3, 3) / 1000
    return v - (v.reshape(-1, 3).min(0) + v.reshape(-1, 3).max(0)) / 2


MESHES = {s: load_mesh(s) for s in SHAPES}


def _rot(a, b):
    """Rotation matrix turning unit vector a onto unit vector b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _yaw(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def pose(shape, rng):
    """Mesh resting on the table (z = 0 is the table), random face down and random turn."""
    face = REST[shape][rng.integers(len(REST[shape]))]
    if face == "side":
        t = rng.uniform(0, 2 * np.pi)
        face = (np.cos(t), np.sin(t), 0)
    tilt = _rot((0, 0, 1), (*rng.normal(0, 0.04, 2), 1))   # a few degrees off: real objects don't sit perfectly
    if rng.random() < 0.1:   # tumbling or held in the hand: any orientation
        tilt = _rot((0, 0, 1), rng.normal(0, 1, 3))
    m = MESHES[shape] @ (tilt @ _yaw(rng.uniform(0, 2 * np.pi)) @ _rot(face, (0, 0, -1))).T
    return m - [0, 0, m[..., 2].min()]


def camera(rng):
    """World-to-camera rotation and camera position: looking at the object from above at a random angle."""
    el, az, dist = rng.uniform(np.radians(25), np.radians(85)), rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.9)
    eye = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    z = -eye / np.linalg.norm(eye)                       # optical axis towards the object
    x = np.cross(z, (0, 0, 1)); x /= np.linalg.norm(x)   # image right
    return np.stack([x, np.cross(z, x), z]), eye + rng.normal(0, 0.01, 3)


def render(shape, rng, bg, size=64, ss=3, cam=None, shift=(0, 0)):
    """Draw one object onto background bg (size x size BGR). Returns the image and the object's box (x, y, w, h).
    cam / shift: reuse a camera and move the object (pixels), to put a neighbour into the same picture."""
    m = pose(shape, rng)
    R, eye = cam or camera(rng)
    hsv = np.uint8([[[rng.integers(0, 180), rng.integers(90, 256), rng.integers(110, 256)]]])
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(float)
    light = np.array([*rng.normal(0, 0.6, 2), 1.0]); light /= np.linalg.norm(light)
    ambient = rng.uniform(0.25, 0.6)
    pc = (m.reshape(-1, 3) - eye) @ R.T                          # camera frame
    f = rng.uniform(0.25, 0.4) * size * ss * pc[:, 2].mean() / 0.045   # object spans 25-40 % of the canvas
    c0 = size * ss / 2 + np.asarray(shift) * ss
    uv = (pc[:, :2] / pc[:, 2:] * f + c0).reshape(-1, 3, 2)
    big = cv2.resize(bg, (size * ss, size * ss)).astype(float)
    ground = m.reshape(-1, 3) - np.outer(m.reshape(-1, 3)[:, 2] / light[2], light)   # shadow on the table (z = 0)
    gc = (ground - eye) @ R.T
    hull = cv2.convexHull(np.int32(gc[:, :2] / gc[:, 2:] * f + c0))
    shade = np.ones(big.shape[:2]); cv2.fillConvexPoly(shade, hull, rng.uniform(0.45, 0.8))
    big *= cv2.GaussianBlur(shade, (0, 0), ss * 1.5)[..., None]
    normals = np.cross(m[:, 1] - m[:, 0], m[:, 2] - m[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    facing = ((m[:, 0] - eye) * normals).sum(1) < 0               # faces pointing at the camera
    mask = np.zeros(big.shape[:2], np.uint8)
    for tri, n in zip(uv[facing], normals[facing]):
        k = ambient + (1 - ambient) * max(0.0, float(n @ light))
        cv2.fillConvexPoly(big, np.int32(np.round(tri)), (color * k).tolist())
        cv2.fillConvexPoly(mask, np.int32(np.round(tri)), 1)
    img = cv2.resize(np.clip(big, 0, 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(mask)
    return img, (x / ss, y / ss, w / ss, h / ss)


def backgrounds(min_side=96):
    """Patches of real table from the robot head-camera frames (below the bins). Frames in the real shape test set
    (testdata/shapes.json) are left out, so that test only measures photos the network has never seen."""
    import json
    held_out = {i["file"] for i in json.load(open(os.path.join(HERE, "testdata", "shapes.json")))}
    out = []
    for f in glob.glob(os.path.join(HERE, "testdata", "head_*.jpg")):
        if os.path.basename(f) in held_out:
            continue
        im = cv2.imread(f)
        for y in range(260, im.shape[0] - min_side, 48):
            out += [im[y:y + min_side, x:x + min_side] for x in range(0, im.shape[1] - min_side, 64)]
    return out


def crop(img, box, rng=None, out=64):
    """Square crop around a box (x, y, w, h), 1.3x its longer side, the same way at training and on the robot.
    With rng: the box is jittered, as colour detection would get it slightly wrong."""
    x, y, w, h = box
    cx, cy, side = x + w / 2, y + h / 2, 1.3 * max(w, h)
    if rng is not None:
        cx, cy, side = cx + rng.normal(0, 0.06) * w, cy + rng.normal(0, 0.06) * h, side * rng.uniform(0.9, 1.15)
    M = np.float32([[out / side, 0, out / 2 - cx * out / side], [0, out / side, out / 2 - cy * out / side]])
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def sample(rng, bgs, size=160):
    """One training example: (64x64 BGR crop, label)."""
    label = int(rng.integers(3))
    bg = bgs[rng.integers(len(bgs))]
    if rng.random() < 0.5:
        bg = bg[:, ::-1]
    bg, cam = cv2.resize(np.ascontiguousarray(bg), (size, size)), camera(rng)
    if rng.random() < 0.5:   # a neighbouring object partly in the picture, as on a crowded table
        a = rng.uniform(0, 2 * np.pi)
        bg, _ = render(SHAPES[rng.integers(3)], rng, bg, size, cam=cam, shift=rng.uniform(0.3, 0.45) * size * np.array([np.cos(a), np.sin(a)]))
    img, box = render(SHAPES[label], rng, bg, size, cam=cam)
    img = crop(img, box, rng)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    img = np.clip(img * rng.uniform(0.7, 1.3) + rng.normal(0, rng.uniform(1, 8), img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(40, 95))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1), label


if __name__ == "__main__":
    import sys
    rng, bgs = np.random.default_rng(0), backgrounds()
    rows = [np.hstack([cv2.resize(sample(rng, bgs)[0], (128, 128)) for _ in range(8)]) for _ in range(6)]
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "render_examples.jpg", np.vstack(rows))
