"""Teammate's ONNX shape classifier on SIM head images of objects on the table, vs truth; and vs our classifier.

    aalto py /data/team/tools/onnx_check.py --n 6
"""
import argparse
from collections import Counter

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    from aaltonomous.brain.colors import find_objects
    from aaltonomous.brain.shapes import classify
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=120, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc, cal = SceneClient(), Calibration.load()
    pol = ClassicalPolicy(safety=env.filter.cfg, log=lambda m: None)
    conf_onnx, conf_ours, conf_fused, rows = Counter(), Counter(), Counter(), []
    TR = {"triangle": "prism"}
    for seed in range(20, 20 + a.n):
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
        hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
        for _ in range(8):
            obs, _ = env.step(hold)
        bgr = cv2.cvtColor(np.ascontiguousarray(obs["head_rgb"]), cv2.COLOR_RGB2BGR)
        objs = find_objects(bgr)
        shapes = classify(bgr, objs)
        dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
        ours_by_id = {id(d): d.shape for d in dets}
        from aaltonomous.baselines.classical.perception import fuse_onnx_shapes
        fuse_onnx_shapes(dets, obs["head_rgb"], pol.perception.size, pol.cfg["perception"])
        for o in sc.state()["objects"]:
            if not o["active"]:
                continue
            u, v, z = cal.project("head", (np.asarray(o["pos"], float) + [0, 0, 0.02])[None])[0]
            truth = TR.get(o["shape"], o["shape"])
            best = min(range(len(objs)), key=lambda i: (objs[i].cx - u) ** 2 + (objs[i].cy - v) ** 2, default=None)
            if best is not None and np.hypot(objs[best].cx - u, objs[best].cy - v) < 40:
                s, p = shapes[best]
                conf_onnx[(truth, s)] += 1
                onnx = f"{s} ({p:.2f})"
            else:
                conf_onnx[(truth, "not found")] += 1
                onnx = "not found"
            d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"][:2])), default=None)
            ours0 = ours_by_id.get(id(d)) if d is not None else None
            ours = TR.get(ours0, ours0)
            fused = TR.get(getattr(d, "shape", None), getattr(d, "shape", None))
            conf_ours[(truth, ours)] += 1
            conf_fused[(truth, fused)] += 1
            rows.append((seed, o["name"][4:], truth, onnx, ours, "FUSED=" + str(fused)))
        obs, _ = env.step(hold)
    env.close()
    for r in rows:
        print("  ", r)
    def acc(c):
        ok = sum(v for (t, p), v in c.items() if t == p)
        return f"{ok}/{sum(c.values())}"
    print("ONNX (teammate):", acc(conf_onnx), dict(conf_onnx))
    print("ours (top face):", acc(conf_ours), dict(conf_ours))
    print("FUSED          :", acc(conf_fused), dict(conf_fused))


if __name__ == "__main__":
    main()
