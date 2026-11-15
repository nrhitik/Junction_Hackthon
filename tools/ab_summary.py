import json, collections, sys
for k in ("base", "mid", "deep"):
    try:
        rows = [json.loads(l) for l in open(f"/data/work/evals/ab_{k}.jsonl")]
    except FileNotFoundError:
        continue
    v = collections.Counter(r["verdict"] for r in rows)
    by = collections.Counter((r["shape"], r["verdict"]) for r in rows)
    print(f"{k:5s} attempts={len(rows)} {dict(v)}")
    print("      by shape:", dict(sorted(by.items(), key=str)))
