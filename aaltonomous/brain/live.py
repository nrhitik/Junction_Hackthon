"""Live view: detection, shapes and tracking on a webcam or a video, drawn as it runs.

    python -m aaltonomous.brain.live                  # laptop webcam (camera 0)
    python -m aaltonomous.brain.live 1                # another camera
    python -m aaltonomous.brain.live data/g1_pick_cube/cam_high.mp4      # the robot's head-camera video
    python -m aaltonomous.brain.live video.mp4 --save out.mp4            # also write the annotated video

Keys: q = quit, space = pause. Uses the same background Watcher as the robot code (track.py). A lid with shaped
holes (lid.py) is looked for every 5th frame and drawn in green, each hole named (box, shape).
The detector expects the robot's view of the whole table: each object about 0.4 to 3 % of the picture. Hold the
webcam so the table with the objects fills the view from roughly half a metre away and above.
"""
import sys
import time

import cv2

from aaltonomous.brain.colors import BGR
from aaltonomous.brain.lid import draw as draw_lid, find_lid
from aaltonomous.brain.track import Watcher


def draw(img, tracks, hz):
    for t in tracks:
        x, y, w, h = map(int, (t.x, t.y, t.w, t.h))
        cv2.rectangle(img, (x, y), (x + w, y + h), BGR[t.color], 2)
        label = f"{t.id} {t.color} {t.shape or ''}"
        for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
            cv2.putText(img, label, (x, max(12, y - 4)), 0, 0.45, col, th)
    cv2.putText(img, f"detection {hz:.0f} per second, {len(tracks)} objects", (8, img.shape[0] - 10), 0, 0.5,
                (255, 255, 255), 2)
    return img


def main(args):
    save = args[args.index("--save") + 1] if "--save" in args else None
    src = next((a for a in args if not a.startswith("--") and a != save), "0")
    cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    watcher, out, paused, frame, lid, n = Watcher(), None, False, None, None, 0
    try:
        while True:
            if not paused:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != 640:   # the detector is tuned for the head camera's 640 px width
                    frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
                watcher.feed(frame)
                n += 1
                lid = find_lid(frame, T=False) if n % 5 == 1 else lid
            shown = draw_lid(draw(frame.copy(), watcher.objects, watcher.hz), lid)
            for e in watcher.events:
                print(e)
            if save:
                out = out or cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, shown.shape[1::-1])
                out.write(shown)
            cv2.imshow("live detection (q = quit, space = pause)", shown)
            wait = 1 if src.isdigit() or paused else max(1, int(1000 / fps - (time.perf_counter() - t0) * 1000))
            key = cv2.waitKey(wait) & 0xFF
            if key == ord("q"):
                break
            paused ^= key == ord(" ")
    finally:
        watcher.stop()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
