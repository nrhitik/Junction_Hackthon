"""How often is the target still in the hand after release (object high at end of retreat/home)?"""
import json, collections
from aaltonomous.scene.scene import Scene
zc = Scene.load().counter_z
for k in ("base", "mid", "deep"):
    rows = [json.loads(l) for l in open(f"/data/evals/ab_{k}.jsonl")]
    c = collections.Counter(); ex = []
    for r in rows:
        tr = r["track"]
        for ph in ("home", "retreat"):
            if tr.get(ph):
                z = tr[ph]["pos"][2] - zc
                stuck = z > 0.10
                c["stuck" if stuck else "free"] += 1
                if stuck: ex.append((r["seed"], r["shape"], round(z, 2), r["final"]))
                break
        else:
            c["no_track"] += 1
    print(f"{k:5s} attempts={len(rows)} {dict(c)} stuck examples={ex}")
