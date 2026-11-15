"""Teammate's find_lid (head camera) vs the sim's true lid holes. Run in the Insert scene.

    aalto py /data/team/tools/lid_check.py --n 4
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.scene.scene import Scene
    from aaltonomous.brain.lid import find_lid, draw
    env = AaltoEnv(backend="sim", task="insert_shapes", seed=0, time_limit=120, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = Scene.load()
    print(f"scene lid: bin={sc.lid_bin} layout={sc.lid_layout} height={sc.lid_height} lid top z(mat)={sc.lid_z[1]:.3f}")
    truth = {}
    for h in sc.lid_holes():
        cx, cy = h["center"]
        p = sc.mat_to_pelvis(np.array([cx, cy, sc.lid_z[1]]))
        truth[{"triangle": "prism"}.get(h["shape"], h["shape"])] = p
    for k, v in truth.items():
        print(f"  TRUE {k:8s} x={v[0]:.3f} y={v[1]:+.3f} z={v[2]:.3f}")
    for seed in range(a.n):
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
        hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
        for _ in range(8):
            obs, _ = env.step(hold)
        bgr = cv2.cvtColor(np.ascontiguousarray(obs["head_rgb"]), cv2.COLOR_RGB2BGR)
        lid = find_lid(bgr, None)  # None = the design (calibration.yaml) head pose
        if lid is None:
            print(f"seed {seed}: NO LID FOUND")
            continue
        out = [f"box={lid.box} z={lid.z:.3f}"]
        for shape in ("cube", "cylinder", "prism"):
            hh = lid.hole(shape)
            if hh is None:
                out.append(f"{shape}: not in view")
                continue
            t = truth[shape]
            err = 1000 * float(np.hypot(hh.x - t[0], hh.y - t[1]))
            out.append(f"{shape}: ({hh.x:.3f},{hh.y:+.3f}) yaw {np.degrees(hh.yaw):.0f} err {err:.0f}mm")
        print(f"seed {seed}: " + " | ".join(out), flush=True)
        if seed == 0:
            cv2.imwrite("/data/evals/lid_check.jpg", draw(bgr, lid))
    env.close()


if __name__ == "__main__":
    main()
