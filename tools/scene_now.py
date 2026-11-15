from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.scene.scene import Scene
zc = Scene.load().counter_z
sc = SceneClient()
for o in sc.state()["objects"]:
    if o["active"]:
        print(o["name"], "z above counter %.3f" % (o["pos"][2] - zc), "pos", [round(v, 3) for v in o["pos"]])
print([ (o["name"], o["outcome"]) for o in sc.evaluate("sort_fixed")["outcomes"]])
