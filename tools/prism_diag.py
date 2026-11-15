"""Why do prisms get the generic grip in full episodes? Replays the first look of given seeds (sort_fixed layouts),
then for every prism: camera shape vs truth, and the wedge planning outcome (best_pinch_plan candidates + reasons).

    aalto py /data/team/tools/prism_diag.py --seeds 1 2 1004
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 1004])
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=600, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(safety=env.filter.cfg, log=lambda m: None)  # before env.reset (one-time setup)
    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"], "task": "sort_fixed"})
            hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
            for _ in range(6):  # fresh camera frames of THIS layout
                obs, _ = env.step(hold)
            dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])

            def busy(fn, *args, **kw):  # plan in a thread while holding (watchdog: 1 s without commands = stop)
                import threading
                nonlocal_obs = {}
                box = {}
                th = threading.Thread(target=lambda: box.update(v=fn(*args, **kw)), daemon=True)
                th.start()
                while th.is_alive():
                    env.step(hold)
                    th.join(0.01)
                return box["v"]
            objs = [o for o in sc.state()["objects"] if o["active"]]
            print(f"\n######## seed {seed}: {len(objs)} objects, {len(dets)} detections", flush=True)
            for o in objs:
                d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"][:2])), default=None)
                derr = None if d is None else 1000 * float(np.linalg.norm(d.top[:2] - np.asarray(o["pos"][:2])))
                print(f"  {o['name'][4:]:18s} true={o['shape']:8s} camera={getattr(d, 'shape', None)} "
                      f"fit={getattr(d, 'shape_fit', None)} pos_err={derr and round(derr, 1)}mm", flush=True)
                if o["shape"] != "triangle" or d is None:
                    continue
                arm_q, hand_q = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
                b = pol._bin(d)
                if d.shape == "triangle":
                    p, cands = busy(pol.planner.best_pinch_plan, d, arm_q, hand_q, b)
                    for c in cands:
                        print(f"      wedge {c['side']:5s} dir {c['dir_deg']:+7.1f}: " +
                              (f"OK score {c['score']} wrist margin {c['wrist_margin']}" if c["ok"] else f"X {c['why']}"),
                              flush=True)
                    print(f"    -> wedge plan: {'YES ' + p.side if p is not None else 'NONE'}", flush=True)
                full = busy(pol.planner.plan, d, arm_q, hand_q, bin_name=b)
                print(f"    -> planner.plan: ok={full.ok} side={full.side} grip={full.shape} "
                      f"{'' if full.ok else full.reason[:160]}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
