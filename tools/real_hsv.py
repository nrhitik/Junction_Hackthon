"""Real cam_high colours vs the classical baseline's sim-tuned HSV ranges.

    aalto py /data/team/tools/real_hsv.py /data/datasets/g1_pick_cube /data/evals/real_hsv
"""
import glob
import os
import sys

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from aaltonomous.baselines.classical.config import load_config

root, out = sys.argv[1:3]
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
t0 = dict(zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"]))
_eps = [int(x) for x in os.environ.get("EPS", "9,17,0").split(",")]
want = {k: t0[k] + 0.2 for k in _eps}
frames = {}
with av.open(glob.glob(root + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
    for f in c.decode(video=0):
        for ep, t in want.items():
            if ep not in frames and float(f.time) >= t:
                frames[ep] = f.to_ndarray(format="bgr24")
        if len(frames) == len(want):
            break

# patches (x, y) in the 640x480 frame of episode 9's first frame (read off the contact sheet)
P = {"violet cube": (340, 330), "violet cylinder": (340, 460), "orange cube": (410, 440), "yellow cyl": (530, 150),
     "yellow cube": (530, 440), "bin (orange)": (380, 100), "mat": (560, 300)}
img = frames[_eps[0]]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("episode 9 frame 0, median HSV (OpenCV: H 0-180) in 11x11 patches:")
for k, (x, y) in P.items():
    p = hsv[y - 5:y + 6, x - 5:x + 6].reshape(-1, 3)
    print(f"  {k:16s} H={np.median(p[:, 0]):5.0f} S={np.median(p[:, 1]):5.0f} V={np.median(p[:, 2]):5.0f}")

import json, os
cfg = load_config(None, json.loads(os.environ.get("AALTO_OVERRIDES", "{}")))
print("hsv:", cfg["perception"]["hsv"])
for ep, img in frames.items():
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vis = img.copy()
    for col, bgr in (("red", (0, 0, 255)), ("yellow", (0, 255, 255)), ("blue", (255, 0, 0))):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in cfg["perception"]["hsv"][col]:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        print(f"ep{ep} baseline mask {col:6s}: {int((m > 0).sum()):6d} px")
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, [c for c in cnts if cv2.contourArea(c) > 120], -1, bgr, 2)
    cv2.imwrite(f"{out}_ep{ep}.jpg", cv2.resize(vis, (480, 360)))
