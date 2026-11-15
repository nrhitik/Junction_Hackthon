"""Train from SUCCESSES + FAILURES (team):
  moves:   imitate the inserted stages (weight 1); failed stages only their early lining-up steps (weight 0.2),
           never their last `bad_tail` steps (what led to the failure)
  release: final state of an inserted stage = "release here" (1); final state of a failed stage = "do NOT release
           here" (0, up-weighted); other steps = 0. The expert never released by itself -> this is the only signal.

    aalto py /data/team/il_insert/train_bc2.py --demos /data/il/cube/demos --out /data/il/cube/bc_pos.pt
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
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--fail_w", type=float, default=0.2)
    ap.add_argument("--bad_tail", type=int, default=3)
    a = ap.parse_args()
    outcome = {}
    for d in a.demos:
        f = os.path.join(d, "stages.jsonl")
        for line in open(f):
            s = json.loads(line)
            outcome[os.path.basename(s["file"])] = s.get("outcome")
    I, L, Y, WM, R, WR = [], [], [], [], [], []
    n_ok = n_bad = 0
    for f in sorted(sum((glob.glob(os.path.join(d, "stage_*.npz")) for d in a.demos), [])):
        oc = outcome.get(os.path.basename(f))
        if oc is None:
            continue
        z = np.load(f)
        T = len(z["label"])
        good = oc == "inserted"
        n_ok += good
        n_bad += not good
        wm = np.ones(T) if good else np.r_[np.full(max(T - a.bad_tail, 0), a.fail_w), np.zeros(min(a.bad_tail, T))]
        r = np.zeros(T)
        wr = np.ones(T)
        r[-1] = 1.0 if good else 0.0
        wr[-1] = 1.0  # rebalanced below
        I.append(z["img"]); L.append(low_scale(z["low"])); Y.append(z["label"]); WM.append(wm); R.append(r)
        WR.append(wr)
        WR[-1] = np.r_[wr[:-1], (-1.0 if good else -2.0)]  # marker: -1 final good, -2 final bad
    I, L, Y, WM, R, WR = map(np.concatenate, (I, L, Y, WM, R, WR))
    fg, fb = WR == -1.0, WR == -2.0
    WR[:] = 0.3                      # ordinary steps: "not yet"
    WR[fg] = max(fb.sum(), 1) * 1.0  # each success final balances all failure finals
    WR[fb] = 3.0                     # failure finals: strong "do NOT release here"
    n = len(Y)
    print(f"stages: {n_ok} inserted, {n_bad} failed; {n} steps; move-weighted steps {int((WM > 0).sum())}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T_ = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    I_ = torch.as_tensor(I, device=dev)
    L_, Y_, WM_, R_, WR_ = map(T_, (L, Y, WM, R, WR))
    net = BCNet(L.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        for k in range(0, n, 128):
            b = perm[k:k + 128]
            sx, sy = rng.integers(-3, 4, 2)
            img = torch.roll(I_[b], shifts=(int(sy), int(sx)), dims=(1, 2))
            d, r = net(img, L_[b] + 0.05 * torch.randn_like(L_[b]))
            lm = (((d - Y_[b, :3]) / MAX_STEP) ** 2).sum(1) * WM_[b]
            lr = F.binary_cross_entropy_with_logits(r, R_[b], reduction="none") * WR_[b]
            loss = lm.sum() / WM_[b].sum().clamp(min=1) + 0.3 * lr.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                d, r = net(I_, L_)
                p = torch.sigmoid(r)
                m = WM_ > 0.5
                err = (1000 * (d[m] - Y_[m, :3]).norm(dim=1)).mean().item()
                pg = p[torch.as_tensor(fg, device=dev)].mean().item()
                pb = p[torch.as_tensor(fb, device=dev)].mean().item()
                po = p[torch.as_tensor(~(fg | fb), device=dev)].mean().item()
            print(f"epoch {ep:3d}  success-move err {err:.2f} mm/step  P(release): success-final {pg:.2f} "
                  f"failure-final {pb:.2f} other {po:.2f}", flush=True)
    torch.save({"state": net.state_dict(), "n_low": L.shape[1], "release_thr": 0.5,
                "note": "successes imitated, failures = do-not-release + low-weight early moves"}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
