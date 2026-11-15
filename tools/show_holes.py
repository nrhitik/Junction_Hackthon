import numpy as np
from aaltonomous.scene.scene import Scene
sc = Scene.load()
c = np.asarray(sc.cfg["bins"]["centers"][sc.lid_bin], float)
for h in sc.lid_holes():
    p = sc.mat_to_pelvis(np.r_[h["center"], sc.lid_z[1]])
    print(f"code {h['shape']:8s} rel box centre mm {(1000 * (np.asarray(h['center']) - c)).round(1)}  "
          f"pelvis x fwd {p[0]:.3f} y left {p[1]:+.3f}")
