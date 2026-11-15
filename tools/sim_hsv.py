"""Measure how objects + bins render in the sim head camera (OpenCV HSV), by projecting the true positions."""
import numpy as np, cv2
import ssl  # noqa
from collections import defaultdict
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=120, img_server_ip="127.0.0.1", head_rgbd=True, head_rgbd_port=55560)
sc, cal, scene = SceneClient(), Calibration.load(), Scene.load()
acc = defaultdict(list)
for seed in range(6):
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
    for _ in range(20):
        obs, _ = env.step(hold)
    hsv = cv2.cvtColor(np.ascontiguousarray(obs["head_rgb"]), cv2.COLOR_RGB2HSV)
    for o in sc.state()["objects"]:
        if not o["active"]:
            continue
        top = np.asarray(o["pos"], float) + [0, 0, float(__import__("os").environ.get("DZ", "0.02"))]
        u, v, z = cal.project("head", top[None])[0]
        if z > 0 and 5 < u < 635 and 5 < v < 475:
            px = hsv[max(0, int(v) - 6):int(v) + 7, max(0, int(u) - 6):int(u) + 7].reshape(-1, 3)
            px = px[(px[:, 2] > 70) & (px[:, 1] > 25)]   # object pixels only: not the dark hand, not the grey mat
            if len(px) > 10:
                acc[o["color"]].append(px)
    for name, c in scene.cfg["bins"]["centers"].items():  # bin front wall
        p = scene.mat_to_pelvis(np.array([c[0], c[1] - 0.1, 0.03]))
        u, v, z = cal.project("head", p[None])[0]
        if z > 0 and 5 < u < 635 and 5 < v < 475:
            acc["BIN"].append(hsv[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4].reshape(-1, 3))
env.close()
for k, v in acc.items():
    a = np.vstack(v).astype(float)
    q = np.percentile(a, [5, 50, 95], axis=0).round().astype(int)
    print(f"{k:7s} H {q[0,0]:3d}/{q[1,0]:3d}/{q[2,0]:3d}  S {q[0,1]:3d}/{q[1,1]:3d}/{q[2,1]:3d}  V {q[0,2]:3d}/{q[1,2]:3d}/{q[2,2]:3d}  (p5/p50/p95, n={len(a)})")
