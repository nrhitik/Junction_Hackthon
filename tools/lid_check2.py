"""Depth-based lid hole positions (perception.locate_lid_holes) vs the sim's true holes."""
import numpy as np
import ssl  # noqa
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import locate_lid_holes
env = AaltoEnv(backend="sim", task="insert_shapes", seed=0, time_limit=120, img_server_ip="127.0.0.1", head_rgbd=True, head_rgbd_port=55560)
pol = ClassicalPolicy(safety=env.filter.cfg, log=lambda m: None)
exp = pol._lid_cache
print("expected (= sim truth):", {k: np.round(v["xy"], 3).tolist() for k, v in exp.items()})
for seed in range(4):
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    LOOK = np.array([0.2, 0.7, 0.4, -0.2, 0.0, -0.5, 0.0, 0.2, -0.7, -0.4, -0.2, 0.0, -0.5, 0.0])  # hands out of view
    print("  look pose team-safety:", pol.team_safety.check(LOOK, pol.post["open"]))
    a, h = pol.planner.go_to(obs["arm_q"], obs["hand_q"], LOOK, pol.post["open"])
    for k in range(0, len(a), 10):
        obs, _ = env.step({"arm_q_target": a[k:k + 10], "hand_q_target": h[k:k + 10]})
    hold = {"arm_q_target": LOOK[None].repeat(5, 0), "hand_q_target": pol.post["open"][None].repeat(5, 0)}
    for _ in range(10):
        obs, _ = env.step(hold)
    import cv2; cv2.imwrite(f"/data/evals/lid_frame_{seed}.jpg", cv2.cvtColor(obs["head_rgb"], cv2.COLOR_RGB2BGR))
    holes, rep = locate_lid_holes(obs["head_rgb"], obs["head_depth"], pol.perception.cal, pol.perception.cam, exp, 0.029)
    print(f"seed {seed}:")
    for r in rep:
        print("   ", r)
env.close()
