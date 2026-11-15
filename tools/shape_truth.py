"""Hand labels (visual, 2026-11-14) for the 223 brain.colors detections on drop-2 first frames (eval_shapes.py order).
c=cube y=cylinder p=prism n=not an object (bin/tape piece); ambiguous ones omitted."""
import json, sys
from collections import Counter
L = {}
def put(s, lab):
    for i in s: L[i] = lab
put([3,7,9,14,17,21,23,26,28,29,30,32,34,36,38,39,42,47, 1,5,11,19,24,33,43,44, 48,49,50,52,55,57,59,62,63,64,65,66,69,70,71,75,77,81,82,83,85,86,87,89,90,91,94,95,
     96,98,99,101,103,105,106,109,110,111,114,115,116,118,119,120,122,123,124,125,126,127,129,130,131,133,134,135,136,137,138,141,143,
     144,145,147,148,149,150,153,154,155,157,159,160,162,163,165,166,167,170,172,174,176,179,181,184,185,186,189,190,191,
     194,195,196,199,200,201,203,204,205,208,210,211,212,216,219,220], "cube")
put([4,10,56,60,67,74,80,84,88,93,100,104,108,113,146,151,156,161,168,173,178,183,188,193,198,207,214,218,222], "cylinder")
put([2,8,16,22,27,31,37,41,46,51,53,61,73,79,97,102,112,117,121,177,182,187,192,197,202,206,213,217,221], "prism")
put([0,6,12,13,15,18,20,25,35,40,45,54,58,68,72,76,78,140,142], "none")
rows = json.load(open(sys.argv[1]))
conf = Counter((L[i], r["shape"]) for i, r in enumerate(rows) if i in L)
objs = [(L[i], r) for i, r in enumerate(rows) if i in L and L[i] != "none"]
ok = sum(t == r["shape"] for t, r in objs)
print(f"labelled {len(L)}/{len(rows)}; real objects {len(objs)}, correct {ok} ({ok / len(objs):.1%}); non-objects detected {sum(1 for v in L.values() if v == 'none')}")
print("truth -> predicted:")
for t in ("cube", "cylinder", "prism", "none"):
    print(f"  {t:9s}", {p: conf[(t, p)] for p in ("cube", "cylinder", "prism") if conf[(t, p)]})
hi = [(t, r["shape"], r["p"]) for t, r in objs if t != r["shape"] and r["p"] >= 0.7]
print("confident (>=0.7) mistakes:", len(hi), Counter((t, s) for t, s, _ in hi))
for th in (0.5, 0.7, 0.85):
    kept = [(t, r) for t, r in objs if r["p"] >= th]
    print(f"  conf>={th}: keeps {len(kept)}/{len(objs)}, accuracy {sum(t == r['shape'] for t, r in kept) / max(1, len(kept)):.1%}")
json.dump({str(k): v for k, v in L.items()}, open(sys.argv[1].replace(".json", "_truth.json"), "w"))
