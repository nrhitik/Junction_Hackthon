"""Prism test, scenario 2 (DEV, sim ground truth for scoring only): ROTATE the prism with a push if no comfortable
pinch is reachable, re-detect with the head camera, then grasp + lift.

    aalto py /data/team/tools/rotate_test.py --seed 7 --out /data/evals/prism_rotate.jsonl --set '{json}'

Per prism: camera detection (shape + orientation; sim truth only as a flagged fallback) -> best_pinch_plan over both
hands x 3 pinch directions. If the chosen plan's wrist margin < --good-margin: find the smallest rotation (+-20/40/60
deg) that makes a pinch with margin >= good-margin reachable (fast IK check, both hands), push a corner tangentially
with the closed hand (planner.nudge_plan: counter/bin/team-safety/safety-filter checked), re-detect, repeat (max 2
nudges). Then grasp, lift, hold 1 s, check (truth + camera), put back, home.
"""
import argparse
import json

import numpy as np


def yaw_of(q):
    x, y, z, w = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def rot(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/data/evals/prism_rotate.jsonl")
    ap.add_argument("--set", default=None)
    ap.add_argument("--good-margin", type=float, default=0.3)
    ap.add_argument("--max-nudges", type=int, default=2)
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy

    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=3600, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(overrides=json.loads(a.set) if a.set else {}, safety=env.filter.cfg, log=lambda m: None)
    pl, scene = pol.planner, pol.scene
    zc, size = scene.counter_z, pol.perception.size
    rng = np.random.default_rng(a.seed)
    (x0, x1), (y0, y1) = scene.layout_kwargs()["region"]
    slots = [(x, y) for y in (y0 + 0.01, 0.5 * (y0 + y1) + 0.01) for x in np.linspace(x0 + 0.02, x1 - 0.02, 5)]
    types = [(c, "triangle") for c in ("red", "yellow", "blue") for _ in range(2)]
    order = rng.permutation(len(slots))[:len(types)]
    layout = [{"color": c, "shape": s, "x": round(float(slots[i][0]), 3), "y": round(float(slots[i][1]), 3),
               "yaw": round(float(rng.uniform(-np.pi / 4, np.pi / 4)), 3)} for (c, s), i in zip(types, order)]
    rows = []
    try:
        obs = env.reset()
        sc.reset(layout)

        last = {"a": np.asarray(obs["arm_q"], float).copy(), "h": np.asarray(obs["hand_q"], float).copy()}

        def hold(n):  # hold the last COMMANDED targets (the measured pose differs -> velocity clips)
            return {"arm_q_target": np.repeat(last["a"][None], n, 0), "hand_q_target": np.repeat(last["h"][None], n, 0)}

        def run(arm, hand):
            nonlocal obs
            for k in range(0, len(arm), 10):
                obs, _ = env.step({"arm_q_target": arm[k:k + 10], "hand_q_target": hand[k:k + 10]})
            if len(arm):
                last["a"], last["h"] = np.asarray(arm[-1], float).copy(), np.asarray(hand[-1], float).copy()

        def busy(fn, *args, **kw):
            """run a planner call in a thread while holding (the env watchdog stops after 1 s without commands)"""
            import threading
            nonlocal obs
            box = {}
            th = threading.Thread(target=lambda: box.update(v=fn(*args, **kw)), daemon=True)
            th.start()
            while th.is_alive():
                obs, _ = env.step(hold(3))
                th.join(0.01)
            return box["v"]

        def truth(name):
            return {o["name"]: o for o in sc.state()["objects"]}[name]

        def detect(name, r):
            """camera detection of `name` with triangle orientation; sim truth only as a flagged fallback."""
            nonlocal obs
            obs, _ = env.step(hold(10))
            t = truth(name)
            dets = [d for d in pol.perception.detect(obs["head_rgb"], obs["head_depth"]) if d.color == t["color"]]
            if not dets:
                return None
            d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(t["pos"][:2])))
            if d.shape != "triangle" or not d.grasp_dirs:
                yw = np.radians(yaw_of(t["quat_xyzw"]))
                d.shape = "triangle"
                d.grasp_dirs = [np.array([np.cos(yw + np.pi / 2 + k), np.sin(yw + np.pi / 2 + k)])
                                for k in (0.0, 2 * np.pi / 3, -2 * np.pi / 3)]
                d.centroid_to_pinch = size / 6.0
                r["truth_fallbacks"] = r.get("truth_fallbacks", 0) + 1
            return d

        obs, _ = env.step(hold(20))
        names = [o["name"] for o in sc.state()["objects"] if o["active"]]
        for name in names:
            t0 = truth(name)
            r = {"object": name, "yaw0": round(yaw_of(t0["quat_xyzw"]), 1), "nudges": []}
            det = detect(name, r)
            if det is None:
                r["result"] = "not detected"; rows.append(r); print(r, flush=True); continue
            plan = None
            for it in range(a.max_nudges + 1):
                arm_q, hand_q = last["a"].copy(), last["h"].copy()
                plan, cands = busy(pl.best_pinch_plan, det, arm_q, hand_q, "center", place_xy=det.top[:2])
                ok = [c for c in cands if c["ok"]]
                best = min(ok, key=lambda c: c["score"]) if ok else None
                r[f"plan_{it}"] = (f"{best['side']} dir {best['dir_deg']:+.0f} wrist margin {best['wrist_margin']}"
                                   if best else "no feasible pinch")
                if (best and best["wrist_margin"] >= a.good_margin) or it == a.max_nudges:
                    break
                # find the smallest rotation that makes a comfortable pinch reachable
                Hm = float(np.clip(det.top[2] - zc, 0.5 * size, 1.5 * size))
                zg = zc + 0.5 * Hm
                target = None
                for dl in (20, -20, 40, -40, 60, -60):
                    dirs = [rot(d, dl) for d in det.grasp_dirs]
                    for side in ("left", "right"):
                        q = busy(pl.quick_pinch_quality, det.top[:2], zg, dirs, arm_q, side)
                        if q is not None and q[2] >= a.good_margin:
                            target = (dl, side, q)
                            break
                    if target:
                        break
                if target is None:
                    r["nudges"].append("no rotation within +-60 deg helps")
                    break
                dl, side_after, _ = target
                c = det.top[:2]
                arc = abs(np.radians(dl)) * (2.0 / 3.0) * size  # vertex at 2/3 altitude from the centroid
                nplan, used = None, None
                for d in det.grasp_dirs:  # push one of the 3 corners tangentially
                    v = c + np.asarray(d) * (2.0 / 3.0) * size
                    tdir = np.array([-d[1], d[0]]) if dl > 0 else np.array([d[1], -d[0]])
                    start, end = v - tdir * 0.045, v + tdir * max(0.0, arc - 0.02)
                    for side in (side_after, "right" if side_after == "left" else "left"):
                        p = busy(pl.nudge_plan, side, arm_q, hand_q, start, end, zc + 0.5 * Hm)
                        if p.ok:
                            nplan, used = p, (side, np.round(v, 3).tolist())
                            break
                    if nplan:
                        break
                if nplan is None:
                    r["nudges"].append(f"rotate {dl:+d}: no safe push")
                    break
                yb = yaw_of(truth(name)["quat_xyzw"])
                run(nplan.arm, nplan.hand)
                obs, _ = env.step(hold(10))
                ya = yaw_of(truth(name)["quat_xyzw"])
                moved = 1000 * float(np.linalg.norm(np.asarray(truth(name)["pos"][:2]) - c))
                r["nudges"].append(f"wanted {dl:+d} deg with {used[0]} hand: got {((ya - yb + 180) % 360) - 180:+.0f} deg, "
                                   f"moved {moved:.0f} mm, z {100 * (truth(name)['pos'][2] - zc):.1f} cm")
                det = detect(name, r)
                if det is None:
                    break
            if plan is None or det is None:
                r["result"] = "no safe plan"; rows.append(r); print(r, flush=True); continue
            r["side"], r["finger_err"] = plan.side, plan.finger_err_deg
            ph = {n: (s, e) for n, s, e in plan.phases}
            i_lift = ph["lift"][1]
            run(plan.arm[:i_lift], plan.hand[:i_lift])
            obs, _ = env.step({"arm_q_target": np.repeat(plan.arm[i_lift - 1][None], 30, 0),
                               "hand_q_target": np.repeat(plan.hand[i_lift - 1][None], 30, 0)})
            o = truth(name)
            lifted = o["pos"][2] - zc > 0.05
            P = pol.perception.color_points(obs["head_rgb"], obs["head_depth"], o["color"])
            gp = pol.kin.grasp_pose(plan.side, plan.arm[i_lift - 1])[0]
            r.update(lifted=bool(lifted), z_cm=round(100 * (o["pos"][2] - zc), 1),
                     camera_says="held" if int((np.linalg.norm(P - gp, axis=1) < 0.06).sum()) >= 30 else "not held")
            run(plan.arm[i_lift:], plan.hand[i_lift:])
            obs, _ = env.step(hold(15))
            r["result"] = "LIFTED" if lifted else "FAILED"
            rows.append(r)
            print({k: r[k] for k in r}, flush=True)
        with open(a.out, "w") as f:
            f.writelines(json.dumps(x, default=str) + "\n" for x in rows)
        print(f"\nROTATE TEST: {sum(x.get('result') == 'LIFTED' for x in rows)}/{len(rows)} lifted", flush=True)
        for x in rows:
            print(f"  {x['object'][4:]:18s} -> {x['result']:12s} {x.get('side', ''):5s} first: {x.get('plan_0', '-')}"
                  f" | nudges: {x['nudges'] or 'none'} | final: {x.get('plan_' + str(len([n for n in x['nudges'] if 'wanted' in n])), '-')}",
                  flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
