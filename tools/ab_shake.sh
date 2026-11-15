#!/bin/bash
cd /data/evals; rm -f ab_midshake.jsonl
aalto py /data/team/tools/diag_attempts.py --seeds 0 1 2 3 4 1004 --out /data/evals/ab_midshake.jsonl \
  --set "{\"hand.closed\": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64], \"grasp.release_shake\": 0.3}" 2>&1 \
  | grep --line-buffered -E "^###|Error|Traceback"
echo AB_DONE
