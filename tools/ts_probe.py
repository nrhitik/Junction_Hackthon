"""Probe the team safety layer: speed, ignored pairs at home, and the cross-body plans it now rejects."""
import time, numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.team_safety import TeamSafety
cfg = load_config(None, {})
kin = ArmKinematics(cfg)
post = hand_postures(cfg, kin.spec)
ts = TeamSafety(cfg, kin)
print("pairs:", len(ts.gm.collisionPairs))
home_sim = home_arm(cfg, kin.spec)
home_robot = home_arm(load_config(None, {}, profile="none"), kin.spec)
ts.calibrate([(home_sim, post["open"]), (home_robot, post["open"])])
print("ignored at home (inherent):", ts.ignored)
print("pairs after calibrate:", len(ts.gm.collisionPairs))
t = time.time(); [ts.check(home_sim, post["open"]) for _ in range(50)]; print(f"check: {(time.time() - t) / 50 * 1000:.1f} ms")
print("home sim:", ts.check(home_sim, post["open"]), "| home robot:", ts.check(home_robot, post["open"]))
