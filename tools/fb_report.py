"""Summarise fb_diag.py logs: score, pick success per true shape / grasp mode, feedback-layer accuracy, stuck objects.

    python3 fb_report.py /data/work/evals/fb_full.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import Counter, defaultdict

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    eps = [r for r in rows if "episode" in r]
    att = [r for r in rows if "episode" not in r and r.get("target")]
    guards = [r for r in rows if r.get("guard")]
    print(f"  GUARD stops: {len(guards)}", [g["guard"][:90] for g in guards][:5])
    print(f"\n===== {path}")
    print(f"score {sum(e['correct'] for e in eps)}/{sum(e['objects'] for e in eps)} correct over {len(eps)} episodes; "
          f"{len(att)} pick attempts; mean t {sum(e['t'] for e in eps) / max(1, len(eps)):.0f}s")
    oc = Counter(v for e in eps for v in e["outcomes"].values())
    print("  outcomes:", dict(oc))
    by = defaultdict(Counter)
    for a in att:
        k = (a["true_shape"], a["grasp_mode"] or "generic")
        by[k]["n"] += 1
        by[k]["held"] += a.get("pick_truth") == "held"
    print("  pick success (truth: held after lift) by (true shape, grasp mode):")
    for k, c in sorted(by.items()):
        print(f"    {k[0]:9s} {k[1]:8s} {c['held']}/{c['n']}")
    cm = Counter((a.get("pick_verdict"), a.get("pick_truth")) for a in att if a.get("pick_verdict"))
    print("  verify_pick (verdict, truth):", dict(cm))
    pc = Counter((p["verdict"], p["truth"]) for a in att for p in a.get("place_checks", []))
    print("  verify_place (verdict, truth):", dict(pc))
    shakes = sum(max(0, len(a.get("place_checks", [])) - 1) for a in att)
    print(f"  shakes triggered: {shakes}")
    fe = [a["finger_err"] for a in att if a.get("finger_err") is not None]
    if fe:
        print(f"  finger alignment error: median {sorted(fe)[len(fe) // 2]} deg, max {max(fe)} deg (n={len(fe)})")
    for a in att:
        if a.get("pick_truth") == "held" and a.get("final") not in ("correct",):
            print(f"    held but final={a.get('final')}: seed {a['seed']} {a['true_shape']} {a['color']}->{a['bin']} "
                  f"place={a.get('place_checks')}")
