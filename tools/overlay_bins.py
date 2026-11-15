"""Project the model's bins (scene.yaml) + the logged object positions onto a real head-camera frame."""
import sys
import cv2
import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy

img = cv2.imread(sys.argv[1]); out = sys.argv[2]
pol = ClassicalPolicy(log=lambda m: None)
cal, cam, sc = pol.perception.cal, pol.perception.cam, pol.scene
H, W = img.shape[:2]
print("frame", W, "x", H, "| calibration image size:", getattr(cam, "width", "?"), getattr(cam, "height", "?"))
b = sc.cfg["bins"]
half = np.asarray(b["outer_size"], float) / 2
ztop = sc.counter_z + float(b["wall_height"])
def P(p):
    u, v, z = cal.project(cam, np.asarray(p, float))
    return int(round(u)), int(round(v)), z
for name, c in b["centers"].items():
    c = np.asarray(c, float)
    for zz, col in ((sc.counter_z, (255, 255, 0)), (ztop, (0, 255, 0))):
        pts = [sc.mat_to_pelvis(np.r_[c + s * half * np.array([sx, sy]), 0.0]) for s, sx, sy in
               [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]]
        uv = []
        for p in pts:
            p = np.r_[p[:2], zz]
            u, v, z = P(p)
            uv.append((u, v))
        cv2.polylines(img, [np.array(uv, np.int32)], True, col, 2)
    print(name, "model centre pelvis", sc.mat_to_pelvis(np.r_[c, ztop])[:2].round(3))
objs = {"yellow cube": (0.386, 0.045, 0.070), "orange cube?": (0.378, 0.054, 0.054), "yellow cyl": (0.361, -0.011, 0.069),
        "purple": (0.358, -0.133, 0.067), "orange cyl": (0.343, 0.087, 0.065)}
for k, p in objs.items():
    u, v, z = P(p)
    cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
    cv2.putText(img, k, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    print(f"{k:13s} logged pelvis {p} -> pixel ({u},{v})")
cv2.imwrite(out, img)
