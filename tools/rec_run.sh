#!/bin/bash
# Record the sim screen (:0) during one fb_diag run. usage: rec_run.sh NAME -- <fb_diag args>
NAME=$1; shift; [ "$1" = "--" ] && shift
OUT=/data/work/recordings/${NAME}_$(date +%H%M).mp4
DISPLAY=:0 ffmpeg -loglevel error -y -f x11grab -framerate 15 -video_size 1920x1080 -i :0 \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$OUT" &
FF=$!
sleep 2
docker exec aalto bash -lc "aalto py /data/team/tools/fb_diag.py $* 2>&1 | grep -E \"^###|^\[sort|^   \""
sleep 2
kill -INT $FF; wait $FF 2>/dev/null
ls -la "$OUT"; echo "REC_DONE $OUT"
