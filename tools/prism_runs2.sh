#!/bin/bash
cd /data/evals
TWO='"hand.closed": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], "grasp.point_in_palm": [0.099, -0.084, 0.015], "grasp.depth_grasp_height": true'
echo "=== RUN C: SIDE grasp (approach tilted 70/60/50 deg towards the object, horizontal pinch), 2 fingers, depth height"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --generic --out /data/evals/prism_side.jsonl \
  --set "{$TWO, \"perception.shape.enabled\": false, \"grasp.shape_grasp.enabled\": false, \"grasp.side_grasp\": {\"enabled\": true, \"tilt_deg\": [70, 60, 50], \"approach_dist\": 0.10}, \"ik.max_tilt_deg\": 20}" 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo "=== RUN D: ORIENTATION-ALIGNED pinch (thumb on a flat, index on the opposite apex), 2 fingers, depth height"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --out /data/evals/prism_aligned.jsonl \
  --set "{$TWO, \"perception.shape.enabled\": true, \"grasp.shape_grasp.enabled\": true, \"grasp.shape_profiles.triangle.enabled\": false, \"grasp.shape_profiles.cylinder.enabled\": false, \"grasp.shape_profiles.cube.enabled\": false}" 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo PRISM2_DONE
