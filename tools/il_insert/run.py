"""Run the rung-3 hybrid in the sim: collect expert demos (DART noise) or evaluate a controller.

    # 2nd sim instance (does not touch the main sim): AALTO_INSTANCE=il TASK=Aalto-Insert-G129-Dex3-Joint aalto sim
    AALTO_INSTANCE=il aalto py /data/team/il_insert/run.py --mode expert --noise 0.004 --seeds 0 1 2 --out /data/il/demos
    AALTO_INSTANCE=il aalto py /data/team/il_insert/run.py --mode classical --seeds 100 101   # cut-free baseline
    AALTO_INSTANCE=il aalto py /data/team/il_insert/run.py --mode learned --model /data/il/bc.pt --seeds 100 101

Prints per final stage: shape, steps, release offset, and the object's outcome (inserted / forced / ...).
Results appended to <out>/results.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expert", "classical", "learned"], default="expert")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="/data/il/demos")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--one-each", action="store_true", default=True)
    ap.add_argument("--shape", default=None, help="team: ONE object of this shape per episode (cube/cylinder/triangle)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import aaltonomous.interface.env as E
    if a.shape:  # team: one object of one shape (train / test the final stage shape by shape)
        _orig_s = E.sample_layout

        def _one(rng, **kw):
            lay = _orig_s(rng, **{**kw, "n_min": 1, "n_max": 1})
            lay[0]["shape"] = a.shape
            return lay
        E.sample_layout = _one
    elif a.one_each:   # one cube, one cylinder, one prism (as the team's fb_diag --one-each)
        _orig = E.sample_layout

        def _one_each(rng, **kw):
            lay = _orig(rng, **{**kw, "n_min": 3, "n_max": 3})
            for o, s in zip(lay, rng.permutation(["cube", "cylinder", "triangle"])):
                o["shape"] = str(s)
            return lay
        E.sample_layout = _one_each
    import ssl  # noqa: F401
    from aaltonomous.sim.evaluate import apply_instance_env
    apply_instance_env()
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from hybrid import HybridInsertPolicy, ExpertController, Recorder

    env = AaltoEnv(backend="sim", task="insert_shapes", seed=0, time_limit=a.time_limit, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    sc = SceneClient()
    ctrl, rec = None, None
    if a.mode == "expert":
        ctrl, rec = ExpertController(sc, noise_m=a.noise, rng=np.random.default_rng(1234)), Recorder(a.out)
    elif a.mode == "learned":
        from learned import LearnedController
        ctrl = LearnedController(a.model)
    log = (lambda m: print("   LOG " + m, flush=True)) if os.environ.get("IL_LOG") else (lambda m: None)
    pol = HybridInsertPolicy(controller=ctrl, recorder=rec, safety=env.filter.cfg, threaded=True,
                             time_limit=a.time_limit, log=log)
    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"], "task": "insert_shapes"})
            n0 = len(pol.final_log)
            t0 = time.time()
            ne = len(pol.events)
            while True:
                obs, info = env.step(pol.act(obs))
                if os.environ.get("IL_EVENTS"):  # team: live event timeline
                    for tt, m in pol.events[ne:]:
                        print(f"   EV [t={tt if tt is None else round(tt, 1)} wall={time.time() - t0:.0f}s] {m[:220]}", flush=True)
                    ne = len(pol.events)
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            ev = sc.evaluate("insert_shapes")
            out = {o["name"]: o.get("outcome") for o in ev["outcomes"]}
            stages = pol.final_log[n0:]
            for s in stages:
                s["outcome"] = out.get(s.get("target"))
            if rec is not None:
                for s in rec.stages:
                    s.setdefault("outcome", out.get(s.get("target")))
                with open(os.path.join(a.out, "stages.jsonl"), "a") as f:
                    for s in rec.stages:
                        f.write(json.dumps({**s, "seed": seed}, default=str) + "\n")
                rec.stages = []
            res = {"mode": a.mode, "seed": seed, "score": ev.get("score"), "outcomes": out, "stages": stages,
                   "t": round(info.get("t", 0), 1), "wall_s": round(time.time() - t0)}
            with open(os.path.join(a.out, "results.jsonl"), "a") as f:
                f.write(json.dumps(res, default=str) + "\n")
            st = "; ".join(f"{s['shape']}:{s['outcome']}@{s['release_rel_mm']}({s['steps']})" for s in stages)
            print(f"### {a.mode} seed {seed}: score {ev.get('score')} {out} | final stages: {st}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
