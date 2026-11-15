#!/bin/bash
# Record the sim screen (:0, what VNC shows) during one task-1 episode. usage: rec_task1.sh SEED
SEED=${1:-2}
OUT=/data/work/recordings/task1_seed${SEED}_$(date +%H%M).mp4
DISPLAY=:0 ffmpeg -loglevel error -y -f x11grab -framerate 15 -video_size 1920x1080 -i :0 \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$OUT" &
FF=$!
sleep 2
docker exec aalto bash -lc "aalto py /data/team/tools/fb_diag.py --task sort_fixed --seeds $SEED --out /data/evals/rec_task1.jsonl 2>&1 | grep -E \"^###|^   outcomes|final check\""
sleep 2
kill -INT $FF; wait $FF 2>/dev/null
ls -la "$OUT"; echo "REC_DONE $OUT"
