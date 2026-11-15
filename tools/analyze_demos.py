"""Analyse real teleop demos (LeRobot v3.0): which hand grasps, closed posture, closing speed, arm speeds.

    aalto py /data/team/tools/analyze_demos.py /data/datasets/g1_pick_cube
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
names = info["features"]["action"]["names"][0]
df = pq.read_table(next((root / "data").rglob("*.parquet"))).to_pydict()
eps = pq.read_table(next((root / "meta/episodes").rglob("*.parquet")))
print("episode meta columns:", [c for c in eps.column_names if not c.startswith("stats/")][:40])
print("frame columns:", list(df.keys()))
print("action names:", names)

S = np.asarray(df["observation.state"], float)
A = np.asarray(df["action"], float)
E = np.asarray(df["episode_index"])
fps = info["fps"]
# DDS hand indices in the 28-vector
L = slice(14, 21)   # left: thumb0,1,2, middle0,1, index0,1
R = slice(21, 28)   # right: thumb0,1,2, index0,1, middle0,1

rows = []
for e in np.unique(E):
    m = E == e
    s, a = S[m], A[m]
    n = len(s)
    out = {"ep": int(e), "dur_s": round(n / fps, 1)}
    for side, sl in (("L", L), ("R", R)):
        h = a[:, sl]
        span = np.abs(h - h[0]).sum(1)
        out[f"{side}_hand_move"] = round(float(span.max()), 2)
    side = "L" if out["L_hand_move"] > out["R_hand_move"] else "R"
    sl = L if side == "L" else R
    arm = slice(0, 7) if side == "L" else slice(7, 14)
    h_cmd, h_meas = a[:, sl], s[:, sl]
    closure = np.abs(h_cmd - h_cmd[0]).sum(1)
    thr = 0.5 * closure.max()
    t_close = int(np.argmax(closure > thr))
    # the most-closed command plateau
    k = int(np.argmax(closure))
    out.update(side=side, t_close_s=round(t_close / fps, 1),
               open_cmd=np.round(h_cmd[0], 2).tolist(),
               closed_cmd=np.round(h_cmd[k], 2).tolist(),
               closed_meas=np.round(h_meas[k], 2).tolist(),
               # measured lag behind command at max closure = finger blocked by the object (contact)
               block=np.round(h_meas[k] - h_cmd[k], 2).tolist())
    # finger closing speed (command), rad/s
    hv = np.abs(np.diff(h_cmd, axis=0)) * fps
    out["hand_vel_p95"] = round(float(np.percentile(hv.max(1), 95)), 2)
    av = np.abs(np.diff(a[:, arm], axis=0)) * fps
    out["arm_vel_p95"] = round(float(np.percentile(av.max(1), 95)), 2)
    out["arm_vel_max"] = round(float(av.max()), 2)
    # arm tracking lag (command - measured), rad
    out["arm_track_err_p95"] = round(float(np.percentile(np.abs(a[:, arm] - s[:, arm]).max(1), 95)), 3)
    out["arm_q_at_close"] = np.round(s[t_close, arm], 2).tolist()
    rows.append(out)

for r in rows:
    print(json.dumps(r))

def agg(key, side):
    v = np.array([r[key] for r in rows if r["side"] == side])
    return np.round(np.median(v, 0), 2).tolist() if len(v) else None

for side in ("L", "R"):
    n = sum(r["side"] == side for r in rows)
    print(f"\n== {side} hand: {n} episodes")
    if n:
        for key in ("open_cmd", "closed_cmd", "closed_meas", "block", "hand_vel_p95", "arm_vel_p95",
                    "arm_track_err_p95", "t_close_s"):
            print(f"  median {key}: {agg(key, side)}")
