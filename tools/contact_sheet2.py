"""Contact sheet of cam_high frames per episode (start / first contact / end) to verify auto-labels by eye.

    aalto py /data/team/tools/contact_sheet.py /data/datasets/g1_pick_cube /data/datasets/g1_pick_cube_labels.jsonl /data/evals/sheet
"""
import glob
import json
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

root, labels, out = sys.argv[1:4]
L = {r["episode_index"]: r for r in map(json.loads, open(labels))}
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
key = "videos/observation.images.cam_high"
import av
_frames = []  # (t, small BGR frame): the whole video decoded once (AV1: OpenCV cannot decode it)
with av.open(glob.glob(root + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
    for f in c.decode(video=0):
        _frames.append((float(f.time), cv2.resize(f.to_ndarray(format="bgr24"), (320, 240))))
_ts = np.array([t for t, _ in _frames])
print("decoded", len(_frames), "frames")


def frame(t):
    return _frames[int(np.argmin(np.abs(_ts - t)))][1].copy()


rows = []
for i, t0, t1 in zip(e["episode_index"], e[key + "/from_timestamp"], e[key + "/to_timestamp"]):
    r = L[i]
    tc = r.get("release_s") or r.get("first_contact_s")
    ts = [t0 + 0.2, t0 + (tc if tc is not None else 0.5 * (t1 - t0)), t1 - 0.3]
    fr = [frame(t) for t in ts]
    row = np.hstack(fr)
    txt = f"ep{i} {r.get('object', '')}->{r.get('target_box', '')} | {r['label']} {r.get('reason', '')[:34]}"
    cv2.rectangle(row, (0, 0), (960, 22), (0, 0, 0), -1)
    cv2.putText(row, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(row)
for k in range(0, len(rows), 6):
    cv2.imwrite(f"{out}_{k // 6}.jpg", np.vstack(rows[k:k + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", (len(rows) + 5) // 6, "sheets")
