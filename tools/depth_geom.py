"""Offline: grasp height + lowest hand point while closing, for depth 22.5 vs 28 mm below the top, per grip."""
import numpy as np
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.perception import Detection
for dft in (None, 0.028):
    ov = {"grasp.grasp_depth_from_top": dft} if dft else {"grasp.grasp_depth_from_top": 0}
    pol = ClassicalPolicy(overrides=ov, log=lambda m: None)
    pl, sc = pol.planner, pol.scene
    zc, H = sc.counter_z, pl.height
    print(f"\n==== grasp depth setting: {'28 mm below top' if dft else 'height_frac 0.5 (22.5 mm below top)'}")
    for shape in ("cube", "cylinder", "triangle"):
        p = np.array([0.30, 0.12])
        det = Detection("yellow", np.r_[p, zc + 0.048], 500, 500, "depth")   # camera measures ~48 mm tall
        det.shape = shape
        if shape == "cube":
            det.grasp_dirs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]
        elif shape == "triangle":
            det.grasp_dirs = [np.array([np.cos(a), np.sin(a)]) for a in (0.0, 2.09, -2.09)]
        plan = pl.plan(det, pol.home.copy(), pol.post["open"].copy(), bin_name="left")
        if not plan.ok:
            print(f"  {shape:9s} NO PLAN: {plan.reason[:120]}")
            continue
        gz = plan.waypoints["grasp"][2] - zc
        ph = {n: (s, e) for n, s, e in plan.phases}
        s, e = ph["close"]
        low = min(float(pl.kin.hand_points(plan.side, plan.arm[k], plan.hand[k])[:, 2].min() - zc) for k in range(s, e))
        print(f"  {shape:9s} grip={plan.shape or 'generic':14s} grasp point {1000*gz:5.1f} mm above table = "
              f"{1000*(0.048-gz):4.1f} mm below the top | lowest hand point while closing {1000*low:5.1f} mm above table")
