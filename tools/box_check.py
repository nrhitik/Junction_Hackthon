"""Sim check: box-front measurement (expect ~0 shift in the sim) + detections after the one-object merge."""
import numpy as np
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.baselines.classical.policy import ClassicalPolicy

env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=120, img_server_ip="127.0.0.1",
               head_rgbd=True, head_rgbd_port=55560)
sc = SceneClient()
pol = ClassicalPolicy(safety=env.filter.cfg, threaded=False, log=lambda m: None)
for seed in (1000, 1001, 3, 7):
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    hold = {"arm_q_target": obs["arm_q"][None].repeat(5, 0), "hand_q_target": obs["hand_q"][None].repeat(5, 0)}
    for _ in range(6):
        obs, _ = env.step(hold)
    pol.reset({"instruction": obs["instruction"], "task": "sort_fixed"})
    pol.arm_cmd, pol.hand_cmd = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
    n0 = len(pol.events)
    pol._look(obs)
    true = [o for o in sc.state()["objects"] if o["active"]]
    ev = [m for _, m in pol.events[n0:] if "box front" in m]
    print(f"seed {seed}: {ev[0] if ev else 'no box event'} | true objects {len(true)}, detected {len(pol.last_detections)}: "
          + ", ".join(f"{d.color} {d.shape}" for d in pol.last_detections), flush=True)
env.close()
