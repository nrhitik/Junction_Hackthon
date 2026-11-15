"""Offline (no sim): plan task-3 lid drops for objects on a grid; report reasons + planning time."""
import time, numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
pol = ClassicalPolicy(log=lambda m: None)
pol.reset({"instruction": "Put each object through the hole of the same shape in the lid.", "task": "insert_shapes"})
pl, sc = pol.planner, pol.scene
print("task", pol.task, "lid bin", sc.lid_bin, "holes", {k: (np.round(v["xy"], 3).tolist(), round(v["z"], 3)) for k, v in pl.insert_holes.items()})
L = sc.cfg["layout"]
pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in np.linspace(*L["region_x"], 3) for y in np.linspace(*L["region_y"], 2)]))
for shape in ("cube", "cylinder", "triangle"):
    for p in pts:
        det = Detection("red", np.r_[p[:2], sc.counter_z + pl.height], 500, 500, "depth")
        det.shape = shape
        if shape == "cube":
            det.grasp_dirs = [np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)]) for k in range(4)]
        elif shape == "triangle":
            det.grasp_dirs = [np.array([np.cos(np.pi / 2 + k), np.sin(np.pi / 2 + k)]) for k in (0, 2 * np.pi / 3, -2 * np.pi / 3)]
            det.centroid_to_pinch = pl.height / 6
        t = time.time()
        plan = pl.plan(det, pol.home.copy(), pol.post["open"].copy(), bin_name=sc.lid_bin, relocate=False)
        print(f"{shape:8s} obj ({p[0]:+.2f},{p[1]:+.2f}) {time.time() - t:5.1f}s {'OK ' + plan.side if plan.ok else 'FAIL ' + plan.reason[:170]}")
