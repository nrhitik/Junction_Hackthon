"""Train the BC final-stage controller on recorded expert stages.

    aalto py /data/team/il_insert/train_bc.py --demos /data/il/cube --out /data/il/cube/bc.pt
Uses every recorded stage (labels are the CLEAN expert action, executed moves had DART noise); stages whose object
ended 'inserted' are weighted x2.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learned import BCNet, low_scale, MAX_STEP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    outcome = {}
    for d in a.demos:
        f = os.path.join(d, "stages.jsonl")
        if os.path.exists(f):
            for line in open(f):
                s = json.loads(line)
                outcome[os.path.basename(s["file"])] = s.get("outcome")
    files = sorted(sum((glob.glob(os.path.join(d, "stage_*.npz")) for d in a.demos), []))
    I, L, Y, W = [], [], [], []
    for f in files:
        z = np.load(f)
        w = 2.0 if outcome.get(os.path.basename(f)) == "inserted" else 1.0
        I.append(z["img"]); L.append(low_scale(z["low"])); Y.append(z["label"]); W.append(np.full(len(z["label"]), w))
    if not files:
        sys.exit("no demos")
    I, L, Y, W = map(np.concatenate, (I, L, Y, W))
    n = len(Y)
    ins = sum(v == "inserted" for v in outcome.values())
    print(f"{len(files)} stages ({ins} inserted), {n} steps, release steps {int(Y[:, 3].sum())}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nv = max(1, n // 8)
    va, tr = idx[:nv], idx[nv:]
    T = lambda x: torch.as_tensor(x, device=dev)
    I_, L_, Y_, W_ = T(I), T(L.astype(np.float32)), T(Y.astype(np.float32)), T(W.astype(np.float32))
    net = BCNet(L.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    pos = max(float(Y[:, 3].mean()), 1e-3)
    best = (np.inf, None)
    for ep in range(a.epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr), device=dev)
        for k in range(0, len(perm), 256):
            b = perm[k:k + 256]
            img = I_[b]
            if True:  # small image shift augmentation (+-3 px)
                sx, sy = rng.integers(-3, 4, 2)
                img = torch.roll(img, shifts=(int(sy), int(sx)), dims=(1, 2))
            d, r = net(img, L_[b] + 0.05 * torch.randn_like(L_[b]))
            ld = (((d - Y_[b, :3]) / MAX_STEP) ** 2).sum(1)
            lr = F.binary_cross_entropy_with_logits(r, Y_[b, 3], pos_weight=torch.tensor(1 / pos - 1, device=dev),
                                                    reduction="none")
            loss = ((ld + 0.5 * lr) * W_[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                d, r = net(I_[va], L_[va])
                err = (1000 * (d - Y_[va, :3]).norm(dim=1)).mean().item()
                acc = ((torch.sigmoid(r) > 0.5).float() == Y_[va, 3]).float().mean().item()
            print(f"epoch {ep:3d}  val move err {err:.2f} mm/step  release acc {acc:.3f}", flush=True)
            if err < best[0]:
                best = (err, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    torch.save({"state": best[1], "n_low": L.shape[1], "release_thr": 0.5, "val_err_mm": best[0]}, a.out)
    print(f"saved {a.out} (best val move err {best[0]:.2f} mm/step)")


if __name__ == "__main__":
    main()
