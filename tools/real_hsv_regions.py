import glob, sys
import av, cv2, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]
e = pq.read_table(glob.glob(root + "/meta/episodes/*/*.parquet")[0]).to_pydict()
t = dict(zip(e["episode_index"], e["videos/observation.images.cam_high/from_timestamp"]))[9] + 0.2
with av.open(glob.glob(root + "/videos/observation.images.cam_high/*/*.mp4")[0]) as c:
    for f in c.decode(video=0):
        if float(f.time) >= t:
            img = f.to_ndarray(format="bgr24"); break
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
R = {"violet cube": (320, 310, 360, 360), "violet cyl": (325, 420, 365, 470), "orange cube": (390, 390, 440, 440),
     "orange cyl": (420, 450, 465, 478), "yellow cyl side": (505, 320, 560, 352), "yellow cyl top": (512, 292, 555, 312),
     "yellow cube": (500, 400, 570, 460), "bin L wall": (40, 170, 220, 220), "bin C wall": (290, 160, 490, 215),
     "bin tape": (60, 232, 180, 248), "mat": (420, 290, 490, 335), "black hand": (180, 260, 260, 300)}
print(f"{'region':16s}  H p5/p50/p95     S p5/p50/p95     V p5/p50/p95")
for k, (x0, y0, x1, y1) in R.items():
    p = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    q = np.percentile(p, [5, 50, 95], axis=0).round().astype(int)
    print(f"{k:16s}  {q[0,0]:3d}/{q[1,0]:3d}/{q[2,0]:3d}      {q[0,1]:3d}/{q[1,1]:3d}/{q[2,1]:3d}      {q[0,2]:3d}/{q[1,2]:3d}/{q[2,2]:3d}")
