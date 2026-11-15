"""DEV ONLY (privileged ground truth): grasp + feedback-layer evaluation in the Aalto sim.

    aalto py /data/team/tools/fb_diag.py --seeds 0 1 2 3 4 --out /data/evals/fb_X.jsonl [--set '{json overrides}']

Per pick attempt: true target (nearest active object of the plan colour to the grasp xy), true shape, grasp mode
used, finger alignment error; at `verify_pick`: the layer's verdict vs the truth (target centre > counter + 5 cm =
held); at `verify_place`: verdict vs truth (target centre > counter + 9 cm = still in the hand). Per episode: score.
"""
import os
import argparse
import json
import re
import time
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="/data/evals/fb.jsonl")
    ap.add_argument("--set", default=None)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--task", default="sort_fixed", choices=["sort_fixed", "sort_language", "insert_shapes"])
    ap.add_argument("--instruction", default=None, help="give the policy THIS instruction instead of the env's")
    ap.add_argument("--mapping", default=None, help='JSON {sim colour: bin} to score against (with --instruction)')
    ap.add_argument("--shape-set", default=None,
                    help="exact objects, e.g. triangle,triangle,cylinder,cube (random positions / yaw / colours)")
    ap.add_argument("--one-each", action="store_true", help="3 objects: one cube, one cylinder, one prism")
    ap.add_argument("--shapes", default=None, help="e.g. cube,triangle: only these shapes in the layouts")
    a = ap.parse_args()
    if a.shape_set:
        import aaltonomous.interface.env as E
        _orig2 = E.sample_layout
        _set = [x.strip() for x in a.shape_set.split(",") if x.strip()]

        def _fixed_set(rng, **kw):
            lay = _orig2(rng, **{**kw, "n_min": len(_set), "n_max": len(_set)})
            for o, s_ in zip(lay, rng.permutation(_set)):
                o["shape"] = str(s_)
            return lay
        E.sample_layout = _fixed_set
    if a.one_each:  # task 3 test layout: exactly 3 objects, one cube, one cylinder, one prism (one per colour)
        import aaltonomous.interface.env as E
        _orig1 = E.sample_layout

        def _one_each(rng, **kw):
            lay = _orig1(rng, **{**kw, "n_min": 3, "n_max": 3})
            for o, s in zip(lay, rng.permutation(["cube", "cylinder", "triangle"])):
                o["shape"] = str(s)
            return lay
        E.sample_layout = _one_each
    if a.shapes:
        import aaltonomous.interface.env as E
        _orig = E.sample_layout
        E.sample_layout = lambda rng, **kw: _orig(rng, **{**kw, "shapes": tuple(a.shapes.split(","))})
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.scene.scene import Scene
    from aaltonomous.baselines.classical.policy import ClassicalPolicy

    env = AaltoEnv(backend="sim", task=a.task, seed=0, time_limit=a.time_limit, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc, scene = SceneClient(), Scene.load()
    zc = scene.counter_z
    pol = ClassicalPolicy(overrides=json.loads(a.set) if a.set else {}, safety=env.filter.cfg, threaded=True,
                          time_limit=a.time_limit, log=(lambda m: print("   LOG " + m, flush=True)) if os.environ.get("FB_LOG") else (lambda m: None))

    def objs():
        return {o["name"]: o for o in sc.state()["objects"] if o["active"]}

    tot = Counter()
    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            instr = a.instruction or obs["instruction"]
            pol.reset({"instruction": instr, "task": a.task})
            print(f"[{a.task}] instruction: {instr!r}  true mapping: {env.mapping}", flush=True)
            seen, att, cur = 0, [], None
            while True:
                obs, info = env.step(pol.act(obs))
                new, seen = pol.events[seen:], len(pol.events)
                for t, msg in new:
                    if msg.startswith("execute"):
                        p = [q for q in pol.plans if q.ok][-1]
                        g = np.asarray(p.waypoints["grasp"][:2])
                        cands = [o for o in objs().values() if o["color"] == p.color]
                        tgt = min(cands, key=lambda o: np.linalg.norm(np.asarray(o["pos"][:2]) - g)) if cands else None
                        cur = {"seed": seed, "t": t, "color": p.color, "side": p.side, "bin": p.target_bin,
                               "target": tgt and tgt["name"], "true_shape": tgt and tgt["shape"],
                               "grasp_mode": p.shape, "finger_err": p.finger_err_deg,
                               "depth_below_top_mm": (round(1000 * float(zc + p.waypoints["measured_height"][2]
                                                                         - p.waypoints["grasp"][2]), 1)
                                                      if "measured_height" in p.waypoints else None),
                               "grasp_err_mm": tgt and round(1000 * float(np.linalg.norm(np.asarray(tgt["pos"][:2]) - g)), 1)}
                        att.append(cur)
                    elif cur is not None and msg.startswith("inhand offset est"):
                        m = re.search(r"est \(([-+\d.]+),([-+\d.]+)\) mm .* gp \(([-\d.]+),([-\d.]+)\)", msg)
                        o = objs().get(cur["target"])
                        if m and o:
                            est = np.array([float(m.group(1)), float(m.group(2))])
                            gp = np.array([float(m.group(3)), float(m.group(4))])
                            tru = 1000 * (np.asarray(o["pos"][:2]) - gp)
                            cur["inhand_est_mm"] = est.round(1).tolist()
                            cur["inhand_true_mm"] = tru.round(1).tolist()
                            cur["inhand_err_mm"] = round(float(np.linalg.norm(est - tru)), 1)
                    elif cur is not None and msg.startswith("inhand:"):
                        cur.setdefault("inhand", []).append(msg[8:90])
                    elif msg.startswith("GUARD"):
                        att.append({"seed": seed, "guard": msg})
                    elif cur is not None and msg.startswith("verify_pick"):
                        o = objs().get(cur["target"])
                        cur["pick_verdict"] = msg.split(": ")[1].split(" ")[0]
                        cur["pick_px"] = re.search(r"= (.*)\)$", msg).group(1)
                        cur["pick_truth"] = "held" if o and o["pos"][2] - zc > 0.05 else "not_held"
                    elif cur is not None and msg.startswith("verify_place"):
                        o = objs().get(cur["target"])
                        stuck = "STUCK" in msg
                        cur.setdefault("place_checks", []).append(
                            {"verdict": "stuck" if stuck else "released",
                             "truth": "stuck" if o and o["pos"][2] - zc > 0.09 else "released",
                             "px": re.search(r"= (.*)\)$", msg).group(1)})
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            if a.mapping:
                ev = sc.evaluate("sort_language", mapping=json.loads(a.mapping))
            else:
                ev = sc.evaluate(a.task, mapping=env.mapping) if a.task == "sort_language" else sc.evaluate(a.task)
            fin = {o["name"]: o["outcome"] for o in ev["outcomes"]}
            for x in att:
                x["final"] = fin.get(x["target"])
            good = "inserted" if a.task == "insert_shapes" else "correct"
            n_ok = sum(v == good for v in fin.values())
            brain = [m for _, m in pol.events if m.startswith("brain")]
            if brain:
                print("   " + brain[0][:300], flush=True)
            print(f"   outcomes {fin}  score {ev.get('score')}", flush=True)
            for tt, m in pol.events:
                if m.startswith(("final check", "instruction parsed", "lid:", "task 3:")):
                    print(f"   [t={tt}] {m[:300]}", flush=True)
            tot["correct"] += n_ok
            tot["objects"] += len(fin)
            tot["attempts"] += len(att)
            tot["wrong"] += sum(v == "wrong" for v in fin.values())
            tot["off"] += sum(v == "off" for v in fin.values())
            with open(a.out, "a") as f:
                for x in att:
                    f.write(json.dumps(x, default=str) + "\n")
                f.write(json.dumps({"episode": seed, "correct": n_ok, "objects": len(fin), "t": round(info["t"], 1),
                                    "outcomes": fin}) + "\n")
            print("   grasp depth below the object top (mm): " + ", ".join(
                f"{x['true_shape']}={x.get('depth_below_top_mm')}" for x in att if x.get("target")), flush=True)
            summ = " ".join(f"{x['true_shape'][:3]}/{(x['grasp_mode'] or 'gen')[:3]}:{x.get('pick_verdict', '-')}"
                            f"{'' if x.get('pick_verdict') is None else '(' + x['pick_truth'][:4] + ')'}"
                            for x in att)
            print(f"### seed {seed}: {n_ok}/{len(fin)} correct, t={info['t']:.0f}s, {len(att)} attempts: {summ}",
                  flush=True)
        print(f"TOTAL {tot['correct']}/{tot['objects']} correct, {tot['attempts']} attempts, wrong={tot['wrong']} "
              f"off={tot['off']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
