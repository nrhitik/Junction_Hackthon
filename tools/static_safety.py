"""Static safety check of hand postures + motion caps against the challenge SafetyFilter config."""
import numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import SafetyConfig
sc = SafetyConfig()
print("SafetyConfig:", {k: v for k, v in vars(sc).items() if not k.startswith("_")})
V = {"base": {}, "mid": {"hand.closed": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64]},
     "deep": {"hand.closed": [0.0, 0.06, 1.08, -1.24, -0.68, -1.24, -0.68]}}
for k, ov in V.items():
    cfg = load_config(None, ov)
    kin = ArmKinematics(cfg); sp = kin.spec
    post = hand_postures(cfg, sp)
    lo, hi = np.asarray(sp.hand_lower), np.asarray(sp.hand_upper)
    worst = min(min((p - lo).min(), (hi - p).min()) for p in map(np.asarray, post.values()))
    print(f"{k:5s} min margin to URDF hand limit (both hands, all postures) = {worst:.3f} rad;"
          f" arm_max_vel={cfg['motion']['arm_max_vel']} hand_max_vel={cfg['motion']['hand_max_vel']}")
