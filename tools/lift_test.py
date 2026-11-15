"""Lift test (DEV, sim ground truth): every object type (3 colours x 3 shapes), one at a time.

    aalto py /data/team/tools/lift_test.py --seed 0 --out /data/evals/lift_test.jsonl

Layout: the 9 types on a 5 x 2 grid in the layout region (8 cm spacing), random yaw (+-45 deg).
Per object: fresh head-camera detection -> hand CENTRED over the detected top-face centre (nearest hand first) ->
descend -> close with the shape's grip (shape_profiles demo grip; cylinder: hand.closed) -> lift -> hold 1 s ->
check (truth: object centre > counter + 5 cm; camera: target pixels at the hand) -> put it back on its spot, open,
retreat, home. Every motion passes the team safety layer (planner) and the challenge SafetyFilter (env).
"""
import argparse
import json
import time

import numpy as np

HZ = 30


def yaw_of(q):
    x, y, z, w = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/evals/lift_test.jsonl")
    ap.add_argument("--set", default=None)
    ap.add_argument("--shapes", default="cube,cylinder,triangle")
    ap.add_argument("--copies", type=int, default=1, help="objects per (colour, shape); the sim has 2")
    ap.add_argument("--generic", action="store_true", help="no shape logic: centred generic grip from the config")
    ap.add_argument("--best-hand", action="store_true",
                    help="hand + pinch direction chosen by planner.best_pinch_plan (orientation-based accuracy score)")
    ap.add_argument("--tripod", action="store_true",
                    help="with --aligned: thumb on the APEX, index + middle on the opposite FLAT (pinch dir reversed)")
    ap.add_argument("--aligned", action="store_true",
                    help="camera shape + orientation (aligned pinch, pinch-midpoint shift kept); sim truth only as a "
                         "flagged fallback when the camera does not classify the prism")
    ap.add_argument("--truth-shape", action="store_true",
                    help="TEST ONLY: shape + orientation from the sim (position still from the camera)")
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy

    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=1800, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(overrides=json.loads(a.set) if a.set else {}, safety=env.filter.cfg, log=lambda m: None)
    pl, scene = pol.planner, pol.scene
    zc = scene.counter_z
    rng = np.random.default_rng(a.seed)
    (x0, x1), (y0, y1) = scene.layout_kwargs()["region"]
    slots = [(x, y) for y in (y0 + 0.01, 0.5 * (y0 + y1) + 0.01) for x in np.linspace(x0 + 0.02, x1 - 0.02, 5)]  # away from bins
    types = [(c, s) for c in ("red", "yellow", "blue") for s in a.shapes.split(",") for _ in range(a.copies)]
    order = rng.permutation(len(slots))[:len(types)]
    layout = [{"color": c, "shape": s, "x": round(float(slots[i][0]), 3), "y": round(float(slots[i][1]), 3),
               "yaw": round(float(rng.uniform(-np.pi / 4, np.pi / 4)), 3)} for (c, s), i in zip(types, order)]
    rows = []
    try:
        obs = env.reset()
        sc.reset(layout)
        hold = lambda n: {"arm_q_target": np.repeat(obs["arm_q"][None], n, 0),  # noqa: E731
                          "hand_q_target": np.repeat(obs["hand_q"][None], n, 0)}
        obs, _ = env.step(hold(20))
        names = [o["name"] for o in sc.state()["objects"] if o["active"]]
        print(f"layout: {len(names)} objects active", flush=True)

        def run(arm, hand):
            nonlocal obs
            for k in range(0, len(arm), 10):
                obs, info = env.step({"arm_q_target": arm[k:k + 10], "hand_q_target": hand[k:k + 10]})
            return obs

        for name in names:
            true = {o["name"]: o for o in sc.state()["objects"]}[name]
            r = {"object": name, "color": true["color"], "shape": true["shape"], "yaw": round(yaw_of(true["quat_xyzw"]), 1)}
            obs, _ = env.step(hold(10))
            dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
            same = [d for d in dets if d.color == true["color"]]
            if not same:
                r["result"] = "not detected"; rows.append(r); print(r, flush=True); continue
            det = min(same, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(true["pos"][:2])))
            if a.aligned:
                r["orient_from"] = "camera"
                if det.shape != true["shape"] or not det.grasp_dirs:
                    yw = np.radians(yaw_of(true["quat_xyzw"]))
                    det.shape = true["shape"]
                    det.grasp_dirs = [np.array([np.cos(yw + np.pi / 2 + k), np.sin(yw + np.pi / 2 + k)])
                                      for k in (0.0, 2 * np.pi / 3, -2 * np.pi / 3)]
                    det.centroid_to_pinch = pol.perception.size / 6.0
                    r["orient_from"] = "SIM TRUTH (camera did not classify)"
                if a.tripod and det.grasp_dirs:
                    # fingers (+pinch dir) on the flat, thumb on the apex: dir = apex -> flat; the pinch middle is
                    # alt/6 from the centroid towards the APEX, i.e. against the new dir
                    det.grasp_dirs = [-np.asarray(d, float) for d in det.grasp_dirs]
                    det.centroid_to_pinch = -pol.perception.size / 6.0
                    r["grip_mode"] = "tripod: thumb on apex, index+middle on flat"
            else:
                det.centroid_to_pinch = 0.0  # CENTRE the grip on the detected top-face centre
            if a.generic:  # hand centred above the detected top-face centre, whatever the orientation
                det.shape, det.grasp_dirs = None, None
            if a.truth_shape:
                yw = np.radians(yaw_of(true["quat_xyzw"]))
                det.shape = true["shape"]
                if det.shape == "triangle":  # flat -> apex (scene.py: apex towards +y in the object frame)
                    det.grasp_dirs = [np.array([np.cos(yw + np.pi / 2 + k), np.sin(yw + np.pi / 2 + k)])
                                      for k in (0.0, 2 * np.pi / 3, -2 * np.pi / 3)]
                elif det.shape == "cube":
                    det.grasp_dirs = [np.array([np.cos(yw + k * np.pi / 2), np.sin(yw + k * np.pi / 2)]) for k in range(4)]
                else:  # cylinder: any direction pinches across the diameter
                    det.grasp_dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.radians(np.arange(0, 360, 30))]
            r.update(det_shape=det.shape, det_err_mm=round(1000 * float(np.linalg.norm(det.top[:2] - np.asarray(true["pos"][:2]))), 1))
            arm_q, hand_q = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
            near = min(("left", "right"), key=lambda s: np.linalg.norm(pol.kin.grasp_pose(s, arm_q)[0][:2] - det.top[:2]))
            plan = None
            if a.best_hand:
                plan, cands = pl.best_pinch_plan(det, arm_q, hand_q, "center", place_xy=det.top[:2])
                r["candidates"] = cands
                if plan is not None:
                    best = min((c for c in cands if c["ok"]), key=lambda c: c["score"])
                    r["chosen"] = f"{best['side']} dir {best['dir_deg']:+.0f}deg score {best['score']} "                                   f"(tilt {best['tilt']}, wrist margin {best['wrist_margin']}, finger err {best['finger_err']})"
                    r["n_ok_candidates"] = sum(c["ok"] for c in cands)
            for side in (() if plan is not None else (near, "right" if near == "left" else "left")):
                p = pl._plan_side_hints(det, side, "center", arm_q, hand_q, place_xy=det.top[:2])
                if p.ok:
                    plan = p; break
                r.setdefault("plan_fail", []).append(f"{side}: {p.reason[:120]}")
            if plan is None:
                r["result"] = "no safe plan"; rows.append(r); print(r, flush=True); continue
            r.update(side=plan.side, grip=plan.shape or "generic", finger_err=plan.finger_err_deg,
                     measured_h_mm=round(1000 * float(plan.waypoints["measured_height"][2]), 1)
                     if "measured_height" in plan.waypoints else None,
                     grasp_z_mm=round(1000 * float(plan.waypoints["grasp"][2] - zc), 1))
            ph = {n: (s, e) for n, s, e in plan.phases}
            i_lift = ph["lift"][1]
            run(plan.arm[:i_lift], plan.hand[:i_lift])
            # hold 1 s at the lift: slipping objects fall now
            last = {"arm_q_target": np.repeat(plan.arm[i_lift - 1][None], 30, 0),
                    "hand_q_target": np.repeat(plan.hand[i_lift - 1][None], 30, 0)}
            obs, _ = env.step(last)
            o = {x["name"]: x for x in sc.state()["objects"]}[name]
            lifted = o["pos"][2] - zc > 0.05
            P = pol.perception.color_points(obs["head_rgb"], obs["head_depth"], true["color"])
            gp = pol.kin.grasp_pose(plan.side, plan.arm[i_lift - 1])[0]
            n_hand = int((np.linalg.norm(P - gp, axis=1) < 0.06).sum())
            r.update(lifted=bool(lifted), z_cm=round(100 * (o["pos"][2] - zc), 1), camera_px=n_hand,
                     camera_says="held" if n_hand >= 30 else "not held")
            run(plan.arm[i_lift:], plan.hand[i_lift:])  # put back, open, retreat, home
            obs, _ = env.step(hold(15))
            o = {x["name"]: x for x in sc.state()["objects"]}[name]
            r["after"] = "back on counter" if abs(o["pos"][2] - zc - 0.0225) < 0.02 else f"z={100 * (o['pos'][2] - zc):.1f}cm"
            r["result"] = "LIFTED" if lifted else "FAILED"
            rows.append(r)
            print({k: r[k] for k in ("object", "result", "side", "chosen", "n_ok_candidates", "grip", "orient_from", "finger_err", "det_err_mm", "measured_h_mm", "grasp_z_mm",
                                     "z_cm", "camera_says", "after") if k in r}, flush=True)
        with open(a.out, "w") as f:
            f.writelines(json.dumps(x, default=str) + "\n" for x in rows)
        ok = sum(x.get("result") == "LIFTED" for x in rows)
        print(f"\nLIFT TEST: {ok}/{len(rows)} lifted", flush=True)
        for x in rows:
            print(f"  {x['color']:6s} {x['shape']:8s} yaw {x['yaw']:+6.1f}  -> {x['result']:13s} "
                  f"{x.get('side', ''):5s} grip={x.get('grip', '-'):14s} detected={x.get('det_shape')} "
                  f"camera={x.get('camera_says', '-')}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
