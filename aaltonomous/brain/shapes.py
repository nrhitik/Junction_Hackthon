"""Shape of each detected object: cube, cylinder or prism (triangular prism). A small network trained only on synthetic images.

    from aaltonomous.brain.shapes import classify
    classify(bgr, find_objects(bgr))      # -> [("cube", 0.97), ("cylinder", 0.88), ...], one per object

    python -m aaltonomous.brain.shapes train [n_images] [epochs]   # render.py images -> shapes.onnx (needs PyTorch)
    python -m aaltonomous.brain.shapes test                        # accuracy on the labelled real photos

On the robot it runs with OpenCV alone (shapes.onnx through cv2.dnn): no PyTorch needed there.
Colour is no clue to shape, so training shuffles the colour channels and the network learns outlines and shading.
"""
import json
import os
import sys

import cv2
import numpy as np

from aaltonomous.brain.render import SHAPES, backgrounds, crop, sample

HERE = os.path.dirname(__file__)
ONNX = os.path.join(HERE, "shapes.onnx")
_net = None


def _blob(crops):
    return (np.float32(crops)[..., ::-1] / 255 - 0.5).transpose(0, 3, 1, 2).copy()   # BGR -> RGB, NCHW, -0.5..0.5


def classify(bgr, objs):
    global _net
    if not objs:
        return []
    _net = _net or cv2.dnn.readNetFromONNX(ONNX)
    _net.setInput(_blob([crop(bgr, (o.x, o.y, o.w, o.h)) for o in objs]))
    z = _net.forward()
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return [(SHAPES[i], float(q[i])) for i, q in zip(p.argmax(1), p)]


def real_set():
    """Labelled real crops: testdata/shapes.json (robot head camera + phone photo), as (crops, labels)."""
    items = json.load(open(os.path.join(HERE, "testdata", "shapes.json")))
    imgs = {f: cv2.imread(os.path.join(HERE, "testdata", f)) for f in {i["file"] for i in items}}
    return np.stack([crop(imgs[i["file"]], i["box"]) for i in items]), np.array([SHAPES.index(i["shape"]) for i in items])


def _make(args):
    seed, n = args
    rng, bgs = np.random.default_rng(seed), backgrounds()
    out = [sample(rng, bgs) for _ in range(n)]
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out])


def train(n=40000, epochs=12):
    import torch
    import torch.nn as nn
    from multiprocessing import Pool
    with Pool() as pool:   # rendering is CPU work: spread it over the cores
        parts = pool.map(_make, [(s, n // 40) for s in range(40)])
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    rx, ry = real_set()
    print(f"rendered {len(x)} images; real test set {len(rx)}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(x) // 256))
    X, Y = torch.from_numpy(_blob(x)), torch.from_numpy(y)
    RX = torch.from_numpy(_blob(rx)).to(dev)
    for ep in range(epochs):
        net.train()
        for i in torch.randperm(len(X)).split(256)[:len(X) // 256]:
            xb = X[i].to(dev)[:, torch.randperm(3)]                  # shuffle colour channels: shape, not colour
            xb = torch.where(torch.rand(len(xb), 1, 1, 1, device=dev) < 0.5, xb, xb.flip(3))   # mirror
            loss = nn.functional.cross_entropy(net(xb), Y[i].to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            acc = (net(RX).argmax(1).cpu().numpy() == ry).mean()
        print(f"epoch {ep + 1}/{epochs}  loss {loss.item():.3f}  real-photo accuracy {acc:.1%}", flush=True)
    torch.onnx.export(net.cpu(), torch.zeros(1, 3, 64, 64), ONNX, input_names=["crop"], output_names=["logits"],
                      dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print("saved", ONNX)


def test():
    global _net
    _net = _net or cv2.dnn.readNetFromONNX(ONNX)
    rx, ry = real_set()
    _net.setInput(_blob(rx))
    pred = _net.forward().argmax(1)
    for k, s in enumerate(SHAPES):
        print(f"{s:9s} {np.sum((pred == k) & (ry == k))}/{np.sum(ry == k)} correct, "
              f"wrongly called {s}: {np.sum((pred == k) & (ry != k))}")
    print(f"overall {np.mean(pred == ry):.1%} of {len(ry)} real objects")
    return pred, ry


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        train(*map(int, sys.argv[2:4]))
    else:
        test()
