"""Per-attempt failure diagnosis for the classical policy in the Aalto sim (DEV ONLY: uses privileged scene state).

    aalto py /data/team/tools/diag_attempts.py --seeds 0 1 2 3 4 5 6 7 --out /data/evals/diag.jsonl

For every pick the policy executes, it finds the targeted object (nearest active object of the planned colour to the
grasp point), records its true pose at the end of every plan phase, and classifies the attempt:
  missed (never lifted), dropped (lifted, then lost before release), released_outside (released but not in the
  target bin), ok. Also logs the perception error (detected top vs true centre, xy, mm).
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="/data/evals/diag.jsonl")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--set", default=None, help="json config overrides")
    ap.add_argument("--policy", default="aaltonomous.baselines.classical.policy:ClassicalPolicy")
    a = ap.parse_args()
    import ssl  # noqa: F401  (before DDS, as in sim/evaluate.py)
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.scene.scene import Scene
    from aaltonomous.baselines.classical.config import load_config

    mod, cls = a.policy.split(":")
    Pol = getattr(__import__(mod, fromlist=[cls]), cls)

    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc, scene = SceneClient(), Scene.load()
    ov = json.loads(a.set) if a.set else {}
    pol = Pol(config=load_config(None, ov), safety=env.filter.cfg, threaded=True, time_limit=a.time_limit)
    zc = scene.counter_z
    lift_z = zc + 0.06  # object centre this high = held above the counter

    def objs():
        return {o["name"]: o for o in sc.state()["objects"] if o["active"]}

    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            attempts, cur, nplans, tick = [], None, 0, 0
            while True:
                obs, info = env.step(pol.act(obs))
                tick += info["ticks"]  # info["ticks"] = ticks in THIS chunk, not cumulative
                started = [e for e in pol.events if e[1].startswith("execute")]
                if len(started) > nplans:  # the policy just started a new pick
                    nplans = len(started)
                    p = [q for q in pol.plans if q.ok][-1]
                    st = objs()
                    g = np.asarray(p.waypoints["grasp"][:2])
                    cands = [o for o in st.values() if o["color"] == p.color]
                    tgt = min(cands, key=lambda o: np.linalg.norm(np.asarray(o["pos"][:2]) - g)) if cands else None
                    det = pol.last_detections
                    dm = min(det, key=lambda d: np.linalg.norm(d.top[:2] - g)) if det else None
                    cur = {"seed": seed, "t": round(info.get("t", 0), 1), "color": p.color, "side": p.side,
                           "bin": p.target_bin, "grasp_xy": g.round(3).tolist(),
                           "target": tgt["name"] if tgt else None, "shape": tgt["shape"] if tgt else None,
                           "true0": np.round(tgt["pos"], 3).tolist() if tgt else None,
                           "quat0": tgt["quat_xyzw"] if tgt else None,
                           "det_xy": np.round(dm.top[:2], 3).tolist() if dm is not None else None,
                           "start_tick": tick - info["ticks"], "phases": [list(ph) for ph in p.phases], "track": {}}
                    cur["perception_err_mm"] = (
                        round(1000 * float(np.linalg.norm(np.asarray(cur["det_xy"]) - np.asarray(cur["true0"][:2]))), 1)
                        if cur["det_xy"] and cur["true0"] else None)
                    attempts.append(cur)
                if cur is not None and cur["target"]:
                    el = tick - cur["start_tick"]
                    for name, s, e in cur["phases"]:
                        if name not in cur["track"] and el >= e:
                            o = objs().get(cur["target"])
                            cur["track"][name] = ({"pos": np.round(o["pos"], 3).tolist(), "speed": round(o["speed"], 3)}
                                                  if o else None)
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            ev = sc.evaluate("sort_fixed")
            fin = {o["name"]: o for o in ev["outcomes"]}
            for at in attempts:
                tr = at["track"]
                names = [n for n, _, _ in at["phases"]]
                zs = [tr[n]["pos"][2] for n in names if tr.get(n)]
                lifted = bool(zs) and max(zs) > lift_z
                rel = next((n for n in names if "release" in n or "open" in n), None)
                pre = [tr[n]["pos"][2] for n in names[:names.index(rel)] if tr.get(n)] if rel else zs
                held = bool(pre) and pre[-1] > lift_z
                at["final"] = fin.get(at["target"], {}).get("outcome")
                at["verdict"] = ("missed" if not lifted else "dropped" if not held
                                 else "ok" if at["final"] == "correct" else "released_outside")
                with open(a.out, "a") as f:
                    f.write(json.dumps(at, default=str) + "\n")
            n_ok = sum(1 for o in ev["outcomes"] if o.get("outcome") == "correct")
            in_hand = [o["name"] for o in objs().values() if o["pos"][2] - zc > 0.10]
            summ = ", ".join(f"{x['shape']}:{x['verdict']}({x['perception_err_mm']}mm)" for x in attempts)
            print(f"### seed {seed}: {n_ok}/{len(ev['outcomes'])} correct, t={info['t']:.0f}s; IN_HAND={in_hand}; attempts: {summ}",
                  flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
