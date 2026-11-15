"""Run the team's brain shape classifier on the first cam_high frame of every episode of a drop; annotated sheets.

    aalto py /data/team/tools/eval_shapes.py /data/datasets/g1_junction_drop_2 /data/evals/shapes
"""
import glob
import json
import sys

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from aaltonomous.brain.colors import find_objects
from aaltonomous.brain.shapes import classify

root, out = sys.argv[1:3]
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
t0 = dict(zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"]))
want = {k: v + 0.2 for k, v in t0.items()}
frames = {}
with av.open(glob.glob(root + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
    for f in c.decode(video=0):
        for ep, t in want.items():
            if ep not in frames and float(f.time) >= t:
                frames[ep] = f.to_ndarray(format="bgr24")
rows, tiles = [], []
for ep in sorted(frames):
    img = frames[ep]
    objs = find_objects(img)
    pred = classify(img, objs)
    vis = img.copy()
    for k, (o, (s, p)) in enumerate(zip(objs, pred)):
        col = (0, 0, 255) if p < 0.7 else (0, 255, 0)
        cv2.rectangle(vis, (o.x, o.y), (o.x + o.w, o.y + o.h), col, 2)
        cv2.putText(vis, f"{k}:{s[:3]} {p:.2f}", (o.x, max(14, o.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(vis, f"{k}:{s[:3]} {p:.2f}", (o.x, max(14, o.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        rows.append({"ep": ep, "k": k, "color": o.color, "box": [o.x, o.y, o.w, o.h], "shape": s, "p": round(p, 3)})
    cv2.putText(vis, f"ep{ep}", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(cv2.resize(vis, (480, 360)))
json.dump(rows, open(out + ".json", "w"))
for i in range(0, len(tiles), 6):
    t = tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6]))
    cv2.imwrite(f"{out}_{i // 6}.jpg", np.vstack([np.hstack(t[j:j + 2]) for j in range(0, 6, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
from collections import Counter
print(len(rows), "objects;", Counter(r["shape"] for r in rows), "; low-confidence (<0.7):", sum(r["p"] < 0.7 for r in rows))
