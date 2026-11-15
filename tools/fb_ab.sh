#!/bin/bash
# usage: fb_ab.sh NAME "SEEDS" JSON overrides
cd /data/evals; rm -f fb_$1.jsonl
aalto py /data/team/tools/fb_diag.py --seeds $2 --out /data/evals/fb_$1.jsonl --set "$3" $4 2>&1 | grep --line-buffered -E "^###|^TOTAL|^\[|^   |Traceback|Error"
echo "DONE $1"
