"""Task 3 debug: one look at seed 5 in the Insert scene: detections, shapes, bins, plan reasons."""
import time, numpy as np
import ssl  # noqa
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.baselines.classical.policy import ClassicalPolicy
env = AaltoEnv(backend="sim", task="insert_shapes", seed=0, time_limit=120, img_server_ip="127.0.0.1", head_rgbd=True, head_rgbd_port=55560)
pol = ClassicalPolicy(safety=env.filter.cfg, log=print)   # BEFORE env.reset: its one-time setup takes a few s
env.rng = np.random.default_rng(5)
obs = env.reset()
pol.reset({"instruction": obs["instruction"], "task": "insert_shapes"})
hold = {"arm_q_target": obs["arm_q"][None].repeat(10, 0), "hand_q_target": obs["hand_q"][None].repeat(10, 0)}
for _ in range(3):
    obs, _ = env.step(hold)
dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
for d in dets:
    print(d, "shape_fit", d.shape_fit, "-> bin", pol._bin(d))
pol.arm_cmd, pol.hand_cmd = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
t = time.time()
res = pol._look(obs)
print("look ->", res[0], f"{time.time() - t:.1f}s", res[1].summary() if res[0] == "plan" else "")
for p in pol.plans:
    print("  plan:", p.color, p.shape, p.ok, p.reason[:200])
env.close()
