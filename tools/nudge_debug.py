"""Offline (no sim): why is no push safe for the right-side prisms of the seed-7 layout? Tries heights/offsets."""
import json, numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
pol = ClassicalPolicy(overrides={"grasp.point_in_palm": [0.099, -0.084, 0.015]}, log=lambda m: None)
pl, sc = pol.planner, pol.scene
zc, size = sc.counter_z, pol.perception.size
rng = np.random.default_rng(7)
(x0, x1), (y0, y1) = sc.layout_kwargs()["region"]
slots = [(x, y) for y in (y0 + 0.01, 0.5 * (y0 + y1) + 0.01) for x in np.linspace(x0 + 0.02, x1 - 0.02, 5)]
types = [(c, "triangle") for c in ("red", "yellow", "blue") for _ in range(2)]
order = rng.permutation(len(slots))[:len(types)]
lay = [(c, slots[i], float(rng.uniform(-np.pi / 4, np.pi / 4))) for (c, s), i in zip(types, order)]
def rot(v, deg):
    c_, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg)); return np.array([c_ * v[0] - s_ * v[1], s_ * v[0] + c_ * v[1]])
for k, (color, (mx, my), yaw) in enumerate(lay):
    P = sc.mat_to_pelvis(np.array([[mx, my, 0.0], [mx + np.cos(yaw), my + np.sin(yaw), 0.0]]))
    c = P[0, :2]
    ex = P[1, :2] - P[0, :2]; ex /= np.linalg.norm(ex)                # object x axis in pelvis
    ey = np.array([-ex[1], ex[0]])
    dirs = [np.cos(np.pi / 2 + a) * ex + np.sin(np.pi / 2 + a) * ey for a in (0, 2 * np.pi / 3, -2 * np.pi / 3)]
    if c[1] > 0.0:
        continue  # left-side prisms were fine
    print(f"== {color} #{k}: centre ({c[0]:+.3f},{c[1]:+.3f})")
    for dl in (20, -20):
        arc = abs(np.radians(dl)) * (2 / 3) * size
        for d in dirs:
            v = c + d * (2 / 3) * size
            t = np.array([-d[1], d[0]]) if dl > 0 else np.array([d[1], -d[0]])
            for zh in (0.5, 0.7):
                for back in (0.045, 0.06):
                    for side in ("right", "left"):
                        p = pl.nudge_plan(side, pol.home.copy(), pol.post["open"].copy(), v - t * back, v + t * max(0, arc - 0.02), zc + zh * size)
                        print(f"   rot {dl:+d} vertex ({v[0]:+.3f},{v[1]:+.3f}) push dir ({t[0]:+.2f},{t[1]:+.2f}) z {zh:.1f}H back {back*100:.1f}cm {side:5s}: "
                              f"{'OK' if p.ok else p.reason[:90]}")
