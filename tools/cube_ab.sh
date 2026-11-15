#!/bin/bash
# Cube grip A (current: middle curls down too) vs B (middle stays half-bent, out of the way): lift + release, stuck?
cd /data/evals
echo "=== A: current cube grip"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes cube --copies 2 --out /data/evals/cube_A.jsonl 2>&1 | grep --line-buffered -E "^LIFT|^  [a-z]|^\{|Traceback|Error"
echo "=== B: middle finger out of the way"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes cube --copies 2 --out /data/evals/cube_B.jsonl \
  --set "{\"hand.closed\": [0.0, -0.02, 1.14, -0.4, -0.4, -1.12, -0.64]}" 2>&1 | grep --line-buffered -E "^LIFT|^  [a-z]|^\{|Traceback|Error"
echo CUBE_AB_DONE
