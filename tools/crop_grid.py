"""Numbered crop grids of every detection in shapes.json (for hand labelling)."""
import glob, json, sys
import av, cv2, numpy as np, pyarrow.parquet as pq
root, js, out = sys.argv[1:4]
rows = json.load(open(js))
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
want = {k: v + 0.2 for k, v in zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"])}
frames = {}
with av.open(glob.glob(root + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
    for f in c.decode(video=0):
        for ep, t in want.items():
            if ep not in frames and float(f.time) >= t:
                frames[ep] = f.to_ndarray(format="bgr24")
tiles = []
for i, r in enumerate(rows):
    x, y, w, h = r["box"]; m = int(0.25 * max(w, h))
    img = frames[r["ep"]]
    c = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
    t = cv2.resize(c, (120, 120))
    cv2.rectangle(t, (0, 0), (120, 18), (0, 0, 0), -1)
    cv2.putText(t, f"{i} {r['shape'][:3]}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(t)
B = 48
for s in range(0, len(tiles), B):
    t = tiles[s:s + B]; t += [np.zeros_like(tiles[0])] * (B - len(t))
    cv2.imwrite(f"{out}_{s // B}.jpg", np.vstack([np.hstack(t[j:j + 8]) for j in range(0, B, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(tiles), "crops")
