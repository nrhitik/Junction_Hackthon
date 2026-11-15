"""Real wrist-camera frames at grasp, carry and release (drop 2 clean episodes) + the head view at release."""
import glob, json, sys
import av, cv2, numpy as np, pyarrow.parquet as pq
root = "/data/datasets/g1_junction_drop_2"
L = {r["episode_index"]: r for r in map(json.loads, open("/data/datasets/g1_junction_drop_2_labels.jsonl"))}
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
EPS = [(23, "left"), (5, "right"), (39, "left")]
def frames(cam, want):
    out = {}
    with av.open(glob.glob(f"{root}/videos/observation.images.{cam}/*/*.mp4")[0]) as c:
        for f in c.decode(video=0):
            for k, t in want.items():
                if k not in out and float(f.time) >= t:
                    out[k] = cv2.resize(f.to_ndarray(format="bgr24"), (320, 240))
            if len(out) == len(want):
                break
    return out
rows = []
for ep, side in EPS:
    r = L[ep]
    t0 = dict(zip(e["episode_index"], e[f"videos/observation.images.cam_{side}_wrist/from_timestamp"]))[ep]
    h0 = dict(zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"]))[ep]
    tc, tr = r.get("first_contact_s") or 1.0, r.get("release_s") or (r["dur_s"] - 1)
    w = frames(f"cam_{side}_wrist", {"grasp": t0 + tc, "carry": t0 + 0.5 * (tc + tr), "release": t0 + tr - 0.1})
    h = frames("cam_high", {"release": h0 + tr - 0.1})
    row = np.hstack([w["grasp"], w["carry"], w["release"], h["release"]])
    cv2.rectangle(row, (0, 0), (1280, 20), (0, 0, 0), -1)
    cv2.putText(row, f"ep{ep} {side} wrist: grasp | carry | RELEASE   || head at release   ({r['task'][:40]})", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    rows.append(row)
cv2.imwrite(sys.argv[1], np.vstack(rows))
