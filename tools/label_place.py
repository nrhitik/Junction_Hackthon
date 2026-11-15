"""Auto-label pick-and-place teleop demos (drop 2+): grasp, lift, where the hand opened, vs the instructed box.

    aalto py /data/team/tools/label_place.py /data/datasets/g1_junction_drop_2 /data/datasets/g1_junction_drop_2_labels.jsonl

Joint data only (no depth / object poses in the drop). Release point = grasp point (FK, pelvis -> mat frame) at the
frame the hand opens after the lift; box = the scene.yaml bin whose inner footprint (+3 cm) contains it.
VERIFY BY VIDEO: the joint data cannot see which object was grasped or whether it actually fell into the box.
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.scene.scene import Scene

root, out = Path(sys.argv[1]), sys.argv[2]
fps = json.loads((root / "meta/info.json").read_text())["fps"]
df = pq.read_table(next((root / "data").rglob("*.parquet"))).to_pydict()
S, A, E = (np.asarray(df["observation.state"], float), np.asarray(df["action"], float), np.asarray(df["episode_index"]))
ep = pq.read_table(glob.glob(str(root / "meta/episodes/*/*.parquet"))[0]).to_pydict()
task = {i: t[0] for i, t in zip(ep["episode_index"], ep["tasks"])}
kin, sc = ArmKinematics(load_config(None, {})), Scene.load()
HAND = {"left": slice(14, 21), "right": slice(21, 28)}
ARM = {"left": slice(0, 7), "right": slice(7, 14)}
DISTAL = [4, 6]  # curl joints (middle_1/index_1 left, index_1/middle_1 right)


def bin_at(p_mat, tol=0.03):
    for n in sc.bin_names:
        lo, hi = sc.bin_inner_box(n)
        if np.all(p_mat[:2] >= lo[:2] - tol) and np.all(p_mat[:2] <= hi[:2] + tol):
            return n
    return None


rows = []
for e in np.unique(E):
    m = E == e
    s, a = S[m], A[m]
    n = len(s)
    instr = task[int(e)]
    want = re.search(r"(left|center|right) box", instr).group(1)
    obj = re.search(r"the (.+?) into", instr).group(1)
    mv = {k: float(np.abs(a[:, sl] - a[0, sl]).sum(1).max()) for k, sl in HAND.items()}
    side = max(mv, key=mv.get)
    r = {"episode_index": int(e), "task": instr, "object": obj, "target_box": want, "dur_s": round(n / fps, 1), "side": side}
    if mv[side] < 0.3:
        r.update(label="idle", reason="hand never moved"); rows.append(r); continue
    hc, hm = a[:, HAND[side]], s[:, HAND[side]]
    clo = np.abs(hc[:, DISTAL]).mean(1)
    closed = clo > 0.5 * clo.max()
    n_close = int(np.sum(np.diff(closed.astype(int)) == 1) + closed[0])
    lag = np.abs(hc[:, DISTAL] - hm[:, DISTAL]).mean(1)
    contact = closed & (lag > 0.12)
    gp = np.array([kin.grasp_pose(side, q)[0] for q in s[:, :14]])  # pelvis frame
    z = gp[:, 2]
    av = np.abs(np.diff(a[:, ARM[side]], axis=0)).max(1) * fps
    terr = float(np.percentile(np.abs(a[:, ARM[side]] - s[:, ARM[side]]).max(1), 95))
    glitch = bool(av.max() > 4.0 or terr > 0.3)
    r.update(n_close=n_close, arm_vel_max=round(float(av.max()), 1), track_err_p95=round(terr, 2), glitch=glitch)
    if not contact.any():
        r.update(label="failed", reason="closed on nothing"); rows.append(r); continue
    ic = np.flatnonzero(contact)
    i0 = int(ic[0])
    lift = float(z[ic].max() - z[i0])
    r.update(first_contact_s=round(i0 / fps, 1), lift_cm=round(100 * lift, 1))
    if lift < 0.04:
        r.update(label="failed", reason=f"no lift ({100 * lift:.1f} cm)"); rows.append(r); continue
    ipk = int(ic[np.argmax(z[ic])])
    opened = np.flatnonzero(~closed[ipk:])
    if not len(opened):
        r.update(label="failed", reason="never released"); rows.append(r); continue
    ir = ipk + int(opened[0])
    p_mat = np.asarray(sc.pelvis_to_mat(gp[ir]))
    box = bin_at(p_mat)
    r.update(release_s=round(ir / fps, 1), release_mat=np.round(p_mat, 3).tolist(),
             release_above_counter_cm=round(100 * (gp[ir][2] - sc.counter_z), 1), release_box=box,
             place_time_s=round((ir - i0) / fps, 1))
    if box is None:
        r.update(label="failed", reason="released outside every box")
    elif box != want:
        r.update(label="wrong_box", reason=f"released over {box}, instructed {want}")
    elif n_close > 1 or glitch:
        r.update(label="messy", reason=", ".join(x for x in (f"{n_close} closes" if n_close > 1 else "",
                                                              "teleop glitch" if glitch else "") if x))
    else:
        r.update(label="success", reason=f"lift {100 * lift:.0f} cm, released over {box}")
    rows.append(r)

with open(out, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
for r in rows:
    print(f"ep {r['episode_index']:2d} {r['side'][0]} {r['object'][:22]:22s}->{r['target_box']:6s} {r['label']:9s} "
          f"{r.get('reason', ''):38s} rel={r.get('release_box')} place={r.get('place_time_s')}s {r['dur_s']}s")
print(Counter(r["label"] for r in rows))
print("hand by target box:", Counter((r["target_box"], r["side"]) for r in rows))
