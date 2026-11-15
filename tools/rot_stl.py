import numpy as np
import trimesh
d = "/opt/aalto/aaltonomous/scene/assets/"
m = trimesh.load(d + "lid_shapes.stl", force="mesh")
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
m.export(d + "lid_shapes_real.stl")
print("rotated STL written", m.bounds.round(2).tolist())
