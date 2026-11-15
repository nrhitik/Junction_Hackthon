"""Where does the time go between a drop and the next pick? One sort_fixed episode, every policy state change and
every look/plan timed (sim time from obs + wall time for the planning thread).

    aalto py /data/team/tools/timing.py --seed 5
"""
import argparse
import time
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=120, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    pol = ClassicalPolicy(safety=env.filter.cfg, threaded=True, log=lambda m: None)
    looks = []
    orig = pol._look

    def timed(obs):
        t0 = time.time()
        r = orig(obs)
        looks.append((float(obs.get("t", 0)), time.time() - t0, r[0], len(pol.plans)))
        return r
    pol._look = timed
    env.rng = np.random.default_rng(a.seed)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "task": "sort_fixed"})
    state, t_state, per_state = None, 0.0, defaultdict(float)
    trace = []
    n_plans_before = 0
    while True:
        st = pol.state
        t = float(obs.get("t", 0))
        if st != state:
            if state is not None:
                per_state[state] += t - t_state
                trace.append((round(t_state, 1), state, round(t - t_state, 2)))
            state, t_state = st, t
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            per_state[state] += float(obs.get("t", 0)) - t_state
            break
    env.close()
    print("\nSTATE TIMELINE (sim t start, state, duration s):")
    for row in trace:
        print("  ", row)
    print("\nLOOKS (sim t, planning WALL s, result, plans tried so far):")
    prev = 0
    for t, w, kind, n in looks:
        print(f"   t={t:6.1f}  plan wall {w:5.2f}s  -> {kind:5s}  candidate plans tried in this look: {n - prev}")
        prev = n
    print("\nTOTAL sim time per state:", {k: round(v, 1) for k, v in sorted(per_state.items(), key=lambda x: -x[1])})
    print("sim real-time factor ~0.4: 1 s of planning WALL time = ~0.4 s of sim time (on the robot: 1 s)")


if __name__ == "__main__":
    main()
