"""DEV ONLY (privileged): shape / pinch-direction perception vs sim ground truth, from the home view.

    aalto py /data/team/tools/shape_check.py --n 30
"""
import argparse
import json
from collections import Counter

import numpy as np


def yaw_of(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--set", default=None)
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=120, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(overrides=json.loads(a.set) if a.set else {}, safety=env.filter.cfg)
    conf, errs, rows = Counter(), {"cube": [], "triangle": []}, []
    try:
        for k in range(a.n):
            env.rng = np.random.default_rng(500 + k)
            obs = env.reset()
            hold = {"arm_q_target": obs["arm_q"][None], "hand_q_target": obs["hand_q"][None]}
            for _ in range(15):
                obs, _ = env.step(hold)
            dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
            objs = [o for o in sc.state()["objects"] if o["active"]]
            for d in dets:
                o = min(objs, key=lambda o: np.linalg.norm(np.asarray(o["pos"][:2]) - d.top[:2]))
                dist = np.linalg.norm(np.asarray(o["pos"][:2]) - d.top[:2])
                conf[(o["shape"], d.shape)] += 1
                yaw = yaw_of(o["quat_xyzw"])
                e = None
                if d.shape == o["shape"] == "cube":
                    a0 = np.arctan2(d.grasp_dirs[0][1], d.grasp_dirs[0][0])
                    e = abs((np.degrees(a0 - yaw) + 45) % 90 - 45)
                elif d.shape == o["shape"] == "triangle":
                    apex = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)])  # scene.py: apex towards +y
                    e = min(np.degrees(np.arccos(np.clip(np.dot(apex, g), -1, 1))) for g in d.grasp_dirs)
                if e is not None:
                    errs[o["shape"]].append(e)
                rows.append(dict(true=o["shape"], det=d.shape, fit=d.shape_fit, pos_err_mm=round(1000 * dist, 1),
                                 yaw_deg=round(np.degrees(yaw), 1), dir_err_deg=None if e is None else round(e, 1)))
        print("confusion (true, detected):", dict(conf))
        n = sum(conf.values())
        ok = sum(v for (t, p), v in conf.items() if t == p)
        print(f"shape accuracy {ok}/{n}")
        for s, v in errs.items():
            if v:
                print(f"{s}: pinch-direction error median {np.median(v):.1f} deg, p90 {np.percentile(v, 90):.1f}, max {max(v):.1f} (n={len(v)})")
        for r in rows:
            if r["true"] != r["det"] or (r["dir_err_deg"] or 0) > 15:
                print("  ", r)
    finally:
        env.close()


if __name__ == "__main__":
    main()
