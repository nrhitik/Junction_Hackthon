"""Grasp/place timing + geometry from the CLEAN (visually verified, label 'success') demo episodes of both drops.

    aalto py /data/team/tools/learn_clean.py

Per episode (active hand): close duration, hold after closing before the lift starts, lift height of the grasp point,
grasp-point height above the counter at release, open duration, closed / open postures (in LEFT order: right hands are
mirrored), arm speed. Grouped by object (cube / prism).
"""
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.scene.scene import Scene

SETS = [("/data/datasets/g1_pick_cube", "/data/datasets/g1_pick_cube_labels.jsonl", "cube"),
        ("/data/datasets/g1_junction_drop_2", "/data/datasets/g1_junction_drop_2_labels.jsonl", None)]
kin, sc = ArmKinematics(load_config(None, {})), Scene.load()
HAND = {"left": slice(14, 21), "right": slice(21, 28)}
ARM = {"left": slice(0, 7), "right": slice(7, 14)}


def to_left(h, side):
    """Right-hand posture (thumb0-2, index0-1, middle0-1) -> left order/sign (thumb0-2, middle0-1, index0-1)."""
    h = np.asarray(h, float)
    return h if side == "left" else -np.r_[h[:3], h[5:7], h[3:5]]


rows = []
for root, lab, obj_default in SETS:
    root = Path(root)
    labels = {r["episode_index"]: r for r in map(json.loads, open(lab))}
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    df = pq.read_table(next((root / "data").rglob("*.parquet"))).to_pydict()
    S, A, E = (np.asarray(df["observation.state"], float), np.asarray(df["action"], float),
               np.asarray(df["episode_index"]))
    for e, r in labels.items():
        if r["label"] != "success":
            continue
        m = E == e
        s, a = S[m], A[m]
        side = r["side"]
        hc = a[:, HAND[side]]
        clo = np.abs(hc[:, [4, 6]]).mean(1)
        lo, hi = np.percentile(clo, 5), clo.max()
        c50 = clo > lo + 0.5 * (hi - lo)
        i50 = int(np.argmax(c50))                                     # closing half way
        i10 = int(np.flatnonzero(clo[:i50 + 1] < lo + 0.1 * (hi - lo))[-1]) if i50 else 0
        i90 = i50 + int(np.argmax(clo[i50:] > lo + 0.9 * (hi - lo)))
        gp = np.array([kin.grasp_pose(side, q)[0] for q in s[:, :14]])
        z = gp[:, 2] - sc.counter_z
        zc = z[i90]
        rise = np.flatnonzero(z[i90:] > zc + 0.01)
        ilift = i90 + int(rise[0]) if len(rise) else None
        opn = np.flatnonzero(~c50[i90:])
        irel = i90 + int(opn[0]) if len(opn) else None
        obj = obj_default or ("prism" if "prism" in r["task"] else "cube")
        av = np.abs(np.diff(a[:, ARM[side]], axis=0)).max(1) * fps
        rows.append(dict(
            set=root.name, ep=int(e), obj=obj, side=side,
            close_s=(i90 - i10) / fps,
            hold_before_lift_s=(ilift - i90) / fps if ilift is not None else np.nan,
            grasp_z_cm=100 * zc,
            lift_cm=100 * (z[i90:irel].max() - zc) if irel else 100 * (z[i90:].max() - zc),
            release_z_cm=100 * z[irel] if irel else np.nan,
            carry_s=(irel - i90) / fps if irel else np.nan,
            closed=to_left(hc[i90], side), meas_closed=to_left(s[i90 + min(15, len(s) - i90 - 1), HAND[side]], side),
            open=to_left(hc[0], side), arm_v95=float(np.percentile(av, 95))))

print(f"{len(rows)} clean episodes")
g = defaultdict(list)
for r in rows:
    g[r["obj"]].append(r)
for obj, rs in g.items():
    print(f"\n== {obj}: {len(rs)} eps ({sum(r['side'] == 'left' for r in rs)} left, {sum(r['side'] == 'right' for r in rs)} right)")
    for k in ("close_s", "hold_before_lift_s", "grasp_z_cm", "lift_cm", "release_z_cm", "carry_s", "arm_v95"):
        v = np.array([r[k] for r in rs], float)
        v = v[np.isfinite(v)]
        print(f"  {k:20s} median {np.median(v):6.2f}  p25 {np.percentile(v, 25):6.2f}  p75 {np.percentile(v, 75):6.2f}")
    for k in ("closed", "meas_closed", "open"):
        print(f"  {k:12s} (left order) median {np.round(np.median([r[k] for r in rs], 0), 2).tolist()}")
for r in rows:
    print(f"  {r['set'][:14]:14s} ep{r['ep']:2d} {r['obj']:5s} {r['side'][0]} close {r['close_s']:.2f}s hold {r['hold_before_lift_s']:.2f}s "
          f"grasp_z {r['grasp_z_cm']:5.1f} lift {r['lift_cm']:5.1f} release_z {r['release_z_cm']:5.1f} carry {r['carry_s']:.1f}s")
