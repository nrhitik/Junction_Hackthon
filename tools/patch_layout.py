p = "/data/aaltonomous-humanoid/aaltonomous/scene/scene.py"
s = open(p).read()
old = '''            # team: the real lid (photo 2026-11-14) is this layout mirrored in depth: circle + square on the NEAR
            # half (towards the robot, -y), triangle on the far half with its apex away from the robot (+y).
            # The sim uses assets/lid_shapes_real.stl for it (scene_cfg.lid_cfg).
            holes = [("cylinder", (cx - xs, cy - yu), 0.0), ("cube", (cx + xs, cy - yu), 0.0),
                     ("triangle", (cx, cy - yt), 0.0)]              # rot 0: apex towards +y'''
new = '''            # team: the real lid = this layout ROTATED 180 deg in the box (operator, 2026-11-14). Seen from the robot:
            # triangle on the far side (apex away from the robot, +y), circle near-left (robot left = +x mat),
            # square near-right (towards the robot middle). Sim: assets/lid_shapes_real.stl = the STL rotated 180.
            holes = [("cylinder", (cx + xs, cy - yu), 0.0), ("cube", (cx - xs, cy - yu), 0.0),
                     ("triangle", (cx, cy - yt), 0.0)]              # rot 0: apex towards +y'''
assert old in s, "layout block not found"
open(p, "w").write(s.replace(old, new))
print("scene.py patched")
