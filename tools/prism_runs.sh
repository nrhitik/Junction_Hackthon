#!/bin/bash
# Prism lift test: hand centred above the prism (no orientation alignment), grasp height from head depth.
cd /data/evals
COMMON='"perception.shape.enabled": false, "grasp.shape_grasp.enabled": false, "grasp.depth_grasp_height": true'
echo "=== RUN A: 2 fingers (thumb vs index, middle straight), grasp point centred between thumb and index"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --generic --out /data/evals/prism_2f.jsonl \
  --set "{$COMMON, \"hand.closed\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"grasp.point_in_palm\": [0.099, -0.084, 0.015]}" 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo "=== RUN B: 3 fingers (thumb vs index + middle), grasp point centred between the fingers"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --generic --out /data/evals/prism_3f.jsonl \
  --set "{$COMMON, \"hand.closed\": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64], \"grasp.point_in_palm\": [0.099, -0.084, 0.0]}" 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo PRISM_DONE
