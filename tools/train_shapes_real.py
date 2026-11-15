"""Retrain the brain shape CNN with hand-labelled REAL crops mixed into the synthetic renders; session cross-validation.

    aalto py /data/team/tools/train_shapes_real.py cv      # 2 folds: train on 2 recording sessions, test on the other 2
    aalto py /data/team/tools/train_shapes_real.py final   # all real crops -> /data/team/models/shapes_real.onnx

Same architecture / input as aaltonomous/brain/shapes.py (64x64 crop, brain.render.crop), so the output .onnx is a
drop-in replacement for brain/shapes.onnx. Real crops: eval_shapes.py detections + shape_truth.py labels ("none" dropped).
Real crops are augmented (crop jitter as in render.crop, colour-channel shuffle, mirror) and oversampled.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from aaltonomous.brain.render import SHAPES, backgrounds, crop, sample

ROOT = "/data/datasets/g1_junction_drop_2"
DET, TRUTH = "/data/evals/shapes.json", "/data/evals/shapes_truth.json"
SESS = [range(0, 16), range(16, 21), range(21, 34), range(34, 48)]   # recording sessions (camera view changes)
FOLDS = [({0, 2}, {1, 3}), ({1, 3}, {0, 2})]                          # (train sessions, test sessions)
NAME = {"cube": "cube", "cylinder": "cylinder", "prism": "prism"}


def sess(ep):
    return next(i for i, r in enumerate(SESS) if ep in r)


def real_items():
    rows, truth = json.load(open(DET)), json.load(open(TRUTH))
    e = pq.read_table(glob.glob(ROOT + "/meta/episodes/*/*.parquet")[0]).to_pydict()
    want = {k: v + 0.2 for k, v in zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"])}
    frames = {}
    with av.open(glob.glob(ROOT + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
        for f in c.decode(video=0):
            for ep, t in want.items():
                if ep not in frames and float(f.time) >= t:
                    frames[ep] = f.to_ndarray(format="bgr24")
    out = []
    for i, r in enumerate(rows):
        lab = truth.get(str(i))
        if lab in NAME:
            out.append((frames[r["ep"]], tuple(r["box"]), SHAPES.index(NAME[lab]), sess(r["ep"])))
    return out


def _make(args):
    seed, n = args
    rng, bgs = np.random.default_rng(seed), backgrounds()
    o = [sample(rng, bgs) for _ in range(n)]
    return np.stack([x[0] for x in o]), np.array([x[1] for x in o])


def _blob(c):
    return (np.float32(c)[..., ::-1] / 255 - 0.5).transpose(0, 3, 1, 2).copy()


def real_crops(items, n_aug, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for img, box, y, _ in items:
        for _ in range(n_aug):
            xs.append(crop(img, box, rng)); ys.append(y)
    return np.stack(xs), np.array(ys)


def train(syn, real_tr, real_te, epochs=12, out=None):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xr, yr = real_crops(real_tr, 60, 0) if real_tr else (np.zeros((0, 64, 64, 3), np.uint8), np.zeros(0, int))
    x, y = np.concatenate([syn[0], xr]), np.concatenate([syn[1], yr])
    tx = np.stack([crop(img, box) for img, box, _, _ in real_te]) if real_te else None
    ty = np.array([yy for _, _, yy, _ in real_te]) if real_te else None
    torch.manual_seed(0)
    block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(x) // 256))
    X, Y = torch.from_numpy(_blob(x)), torch.from_numpy(y)
    for ep in range(epochs):
        net.train()
        for i in torch.randperm(len(X)).split(256)[:len(X) // 256]:
            xb = X[i].to(dev)[:, torch.randperm(3)]
            xb = torch.where(torch.rand(len(xb), 1, 1, 1, device=dev) < 0.5, xb, xb.flip(3))
            loss = nn.functional.cross_entropy(net(xb), Y[i].to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    res = None
    if tx is not None:
        with torch.no_grad():
            pred = net(torch.from_numpy(_blob(tx)).to(dev)).argmax(1).cpu().numpy()
        res = (pred, ty)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.onnx.export(net.cpu(), torch.zeros(1, 3, 64, 64), out, input_names=["crop"], output_names=["logits"],
                          dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    return res


def report(tag, pred, ty):
    s = f"{tag}: {np.mean(pred == ty):.1%} of {len(ty)}"
    for k, n in enumerate(SHAPES):
        m = ty == k
        s += f" | {n} {np.sum(pred[m] == k)}/{m.sum()}"
    s += f" | cube->prism {np.sum((ty == 0) & (pred == 2))}"
    print(s, flush=True)


def baseline_onnx(items):
    net = cv2.dnn.readNetFromONNX("/opt/aalto/aaltonomous/brain/shapes.onnx")
    net.setInput(_blob(np.stack([crop(img, box) for img, box, _, _ in items])))
    return net.forward().argmax(1), np.array([y for *_, y, _ in [(a, b, c, d) for a, b, c, d in items]])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    items = real_items()
    print(f"{len(items)} labelled real crops", flush=True)
    with Pool() as pool:
        parts = pool.map(_make, [(s, 1000) for s in range(40)])
    syn = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    print(f"{len(syn[0])} synthetic renders", flush=True)
    if mode == "cv":
        for tr, te in FOLDS:
            itr = [it for it in items if it[3] in tr]
            ite = [it for it in items if it[3] in te]
            p0, y0 = baseline_onnx(ite)
            report(f"test sessions {sorted(te)} | existing shapes.onnx", p0, y0)
            p1, y1 = train(syn, [], ite)
            report(f"test sessions {sorted(te)} | retrained synthetic only", p1, y1)
            p2, y2 = train(syn, itr, ite)
            report(f"test sessions {sorted(te)} | synthetic + real (sessions {sorted(tr)})", p2, y2)
    else:
        train(syn, items, [], out="/data/team/models/shapes_real.onnx")
        print("saved /data/team/models/shapes_real.onnx")
