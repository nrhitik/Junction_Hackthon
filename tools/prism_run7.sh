#!/bin/bash
cd /data/evals
echo "=== RUN H: ROTATE-THEN-GRASP (push a corner if no comfortable pinch), aligned pinch thumb on flat / index on apex, best hand"
aalto py /data/team/tools/rotate_test.py --seed 7 --out /data/evals/prism_rotate.jsonl \
  --set "{\"hand.closed\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"hand.closed_triangle\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"grasp.point_in_palm\": [0.099, -0.084, 0.015], \"grasp.depth_grasp_height\": true, \"perception.shape.enabled\": true, \"grasp.shape_grasp.enabled\": true, \"grasp.shape_profiles.triangle.enabled\": false, \"grasp.shape_profiles.cylinder.enabled\": false, \"grasp.shape_profiles.cube.enabled\": false}" 2>&1 \
  | grep --line-buffered -E "^\{|^ROTATE|^  [a-z]|Traceback|Error"
echo PRISM7_DONE
