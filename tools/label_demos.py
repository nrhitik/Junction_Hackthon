"""Auto-label teleop demos from joint data (the drop ships no success labels).

    aalto py /data/team/tools/label_demos.py /data/datasets/g1_pick_cube /data/datasets/g1_pick_cube_labels.jsonl

Per episode, for the active hand:
  close events      command closure crosses 50 % of its max (count = regrasps)
  contact           at the most-closed command, distal finger joints lag the command by > 0.12 rad (blocked by an object)
  lift              palm (FK) rises > 4 cm while closed + in contact
  held_to_end       still closed + in contact in the last 10 % of the episode (no release = 'pick up' done)
  glitch            commanded arm velocity > 4 rad/s (teleop jump) or arm tracking error p95 > 0.3 rad
Label: success = contact + lift; messy = success but regrasps > 1 or glitch; failed = no contact or no lift; idle = no hand motion.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics

root, out = Path(sys.argv[1]), sys.argv[2]
fps = json.loads((root / "meta/info.json").read_text())["fps"]
df = pq.read_table(next((root / "data").rglob("*.parquet"))).to_pydict()
S, A, E = (np.asarray(df["observation.state"], float), np.asarray(df["action"], float), np.asarray(df["episode_index"]))
kin = ArmKinematics(load_config(None, {}))
HAND = {"left": slice(14, 21), "right": slice(21, 28)}
ARM = {"left": slice(0, 7), "right": slice(7, 14)}
DISTAL = {"left": [4, 6], "right": [4, 6]}  # middle_1/index_1 (left), index_1/middle_1 (right): curl joints


def palm_z(side, s):
    zs = []
    for q in s[:, :14]:
        zs.append(float(kin.grasp_pose(side, q)[0][2]))
    return np.array(zs)


rows = []
for e in np.unique(E):
    m = E == e
    s, a = S[m], A[m]
    n = len(s)
    mv = {k: float(np.abs(a[:, sl] - a[0, sl]).sum(1).max()) for k, sl in HAND.items()}
    side = max(mv, key=mv.get)
    r = {"episode_index": int(e), "dur_s": round(n / fps, 1), "side": side}
    if mv[side] < 0.3:
        r.update(label="idle", reason="hand never moved")
        rows.append(r); continue
    hc, hm = a[:, HAND[side]], s[:, HAND[side]]
    clo = np.abs(hc[:, DISTAL[side]]).mean(1)              # curl command (rad)
    closed = clo > 0.5 * clo.max()
    edges = np.flatnonzero(np.diff(closed.astype(int)) == 1)
    n_close = len(edges) + int(closed[0])
    lag = np.abs(hc[:, DISTAL[side]] - hm[:, DISTAL[side]]).mean(1)
    contact = closed & (lag > 0.12)
    z = palm_z(side, s)
    lift = 0.0
    if contact.any():
        i0 = int(np.flatnonzero(contact)[0])
        lift = float(z[contact].max() - z[i0])
    tail = slice(int(0.9 * n), n)
    held_end = bool(contact[tail].mean() > 0.5)
    av = np.abs(np.diff(a[:, ARM[side]], axis=0)).max(1) * fps
    terr = np.percentile(np.abs(a[:, ARM[side]] - s[:, ARM[side]]).max(1), 95)
    glitch = bool(av.max() > 4.0 or terr > 0.3)
    r.update(n_close=n_close, contact_frac=round(float(contact.mean()), 2), lift_cm=round(100 * lift, 1),
             held_to_end=held_end, arm_vel_max=round(float(av.max()), 1), track_err_p95=round(float(terr), 2),
             glitch=glitch, first_contact_s=round(float(np.flatnonzero(contact)[0]) / fps, 1) if contact.any() else None)
    if not contact.any():
        r.update(label="failed", reason="closed on nothing (no finger contact)")
    elif lift < 0.04:
        r.update(label="failed", reason=f"contact but no lift ({100 * lift:.1f} cm)")
    elif n_close > 1 or glitch:
        r.update(label="messy", reason=", ".join(x for x in (f"{n_close} closes" if n_close > 1 else "",
                                                              "teleop glitch" if glitch else "") if x))
    else:
        r.update(label="success", reason="contact + lift" + ("" if held_end else ", released before end"))
    rows.append(r)

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
for r in rows:
    print(f"ep {r['episode_index']:2d} {r['label']:8s} {r.get('reason',''):40s} "
          f"closes={r.get('n_close')} lift={r.get('lift_cm')}cm contact={r.get('contact_frac')} "
          f"vmax={r.get('arm_vel_max')} terr={r.get('track_err_p95')} {r['dur_s']}s")
from collections import Counter
print(Counter(r["label"] for r in rows))
