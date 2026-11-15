"""Task 3, one shape at a time, PLAN ONLY (no episode, seconds per layout): spawn one object of a shape, look once from
home, print true vs detected shape and the plan result / reason.

    aalto py /data/team/tools/plan_check.py --shape triangle --seeds 0 1 2 3 4 5 6 7
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=["cube", "cylinder", "triangle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--set", default=None)
    a = ap.parse_args()
    import json
    import aaltonomous.interface.env as E
    orig = E.sample_layout

    def one(rng, **kw):
        lay = orig(rng, **{**kw, "n_min": 1, "n_max": 1})
        lay[0]["shape"] = a.shape
        return lay
    E.sample_layout = one
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env = AaltoEnv(backend="sim", task="insert_shapes", seed=0, time_limit=120, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(overrides=json.loads(a.set) if a.set else {}, safety=env.filter.cfg, threaded=False,
                          time_limit=120, log=lambda m: None)
    ok_n = cls_n = 0
    for seed in a.seeds:
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
        hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
        for _ in range(6):
            obs, _ = env.step(hold)
        pol.reset({"instruction": obs["instruction"], "task": "insert_shapes"})
        pol.arm_cmd, pol.hand_cmd = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
        o = [x for x in sc.state()["objects"] if x["active"]][0]
        pol._lid_tuned = True
        import time as _t
        t0 = _t.time()
        if os.environ.get("PROFILE"):
            import cProfile, pstats, io
            pr = cProfile.Profile(); pr.enable()
            res = pol._look(obs)
            pr.disable(); st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(28)
            print(st.getvalue()[:6000])
        else:
            res = pol._look(obs)
        print(f"   plan time {_t.time() - t0:.1f} s", flush=True)
        dets = pol.last_detections
        d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"][:2]))) if dets else None
        p = pol.plans[-1] if pol.plans else None
        cls_ok = d is not None and d.shape == a.shape
        ok_n += res[0] == "plan"
        cls_n += cls_ok
        fit = d and getattr(d, "shape_fit", None)
        fit = fit and ",".join(f"{k[:3]}{v:.2f}" for k, v in fit.items())
        print(f"seed {seed}: true {o['color']} {a.shape} at ({o['pos'][0]:+.3f},{o['pos'][1]:+.3f}) | "
              f"seen {d and d.shape} [{d and getattr(d, 'shape_src', '')}] fit {fit} | "
              + (f"PLAN {p.side} {p.summary()[:90]} {'relocate' if 'relocat' in p.summary() else ''}"
                 if res[0] == "plan" else f"NO PLAN: {(p.reason if p else 'no detection')[:300]}"), flush=True)
    print(f"### {a.shape}: classified {cls_n}/{len(a.seeds)}, planned {ok_n}/{len(a.seeds)}")
    env.close()


if __name__ == "__main__":
    main()
