"""Where to let go of an object: above its hole in the lid, or a free spot in an open box (robot frame, m).

    lid = find_lid(bgr, T)                          # lid.py, once, before the hand is over the boxes
    d = drop_point("left", "cube", lid=lid)         # Drop(x, y, z, yaw): object centre when the hand opens
    d = drop_point("center", "cube", k=2)           # open box; k = objects already put into that box
    d is None                                        # lid box, but that hole is not in view: look again

Lid box: right above the hole of the object's shape, its bottom 2 cm above the lid; turn the object to `yaw`
(see lid.py: any multiple of 90 deg more for a cube, of 120 deg for a prism, anything for a cylinder).
Open box: spots 5 cm apart around the middle, so objects don't land on each other; z clears the 8 cm walls. yaw None
= any. Box positions come from the organisers' scene file, or `centre` (x, y) if measured.
"""
from collections import namedtuple

import numpy as np

from aaltonomous.brain.calibrate import TABLE_Z, bins
from aaltonomous.brain.locate import SIZE

Drop = namedtuple("Drop", "x y z yaw")
WALL = 0.08                                                       # boxes are 7-8 cm high (measured)
SPOTS = [(0, 0), (0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05)]   # inside the 21.5 cm box, any turn


def drop_point(box, shape=None, lid=None, k=0, centre=None):
    if lid is not None and lid.box == box:
        h = lid.hole(shape)
        return h and Drop(h.x, h.y, lid.z + SIZE / 2 + 0.02, h.yaw)
    x, y = (bins()[box][:, :2].mean(0) if centre is None else np.asarray(centre)) + SPOTS[k % len(SPOTS)]
    return Drop(float(x), float(y), TABLE_Z + WALL + SIZE / 2 + 0.03, None)
