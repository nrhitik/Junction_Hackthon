"""Team 2026-11-14: switch the sim + tasks to the real colours orange / yellow / purple (organisers' update)."""
import os
R = "/data/aaltonomous-humanoid/aaltonomous"
p = f"{R}/scene/scene.yaml"; s = open(p).read()
old = """    red: [200, 30, 30]
    yellow: [235, 200, 20]
    blue: [30, 70, 200]"""
new = """    # team 2026-11-14 (organisers: real colours orange / yellow / purple). RGB chosen to match the REAL head camera:
    # orange objects OpenCV hue ~13 (bins ~5, darker), purple ~112.
    orange: [245, 120, 30]
    yellow: [235, 200, 20]
    purple: [90, 125, 225]"""
assert old in s; s = s.replace(old, new)
old = """  rgb: [230, 110, 30]                         # (default) orange, same for all boxes; take from a venue photo."""
new = """  rgb: [200, 60, 35]                          # team: darker red-orange of the real boxes (head-cam hue ~5, drops 1-2)"""
assert old in s; s = s.replace(old, new); open(p, "w").write(s)
for p in (f"{R}/interface/tasks.py", f"{R}/organisers/tasks.py"):
    s = open(p).read()
    s = s.replace('COLORS = ("red", "yellow", "blue")',
                  'COLORS = ("orange", "yellow", "purple")   # team 2026-11-14: the real colours (organisers)')
    s = s.replace('FIXED_MAPPING = {"red": "left", "yellow": "center", "blue": "right"}',
                  'FIXED_MAPPING = {"yellow": "left", "purple": "center", "orange": "right"}   # team: organisers 2026-11-14')
    s = s.replace('FIXED_INSTRUCTION = ("Put the red objects in the left bin, the yellow objects in the center bin "\n'
                  '                     "and the blue objects in the right bin.")',
                  'FIXED_INSTRUCTION = ("Put the yellow objects in the left bin, the purple objects in the center bin "\n'
                  '                     "and the orange objects in the right bin.")')
    assert '"orange", "yellow", "purple"' in s and "purple objects in the center" in s, p
    open(p, "w").write(s)
p = f"{R}/sim/check_scene.py"; s = open(p).read()
s = s.replace('"red"', '"orange"').replace('"blue"', '"purple"'); open(p, "w").write(s)
print("recolored")
