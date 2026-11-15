#!/bin/bash
cd /data/evals
echo "=== RUN G2 (signed pinch IK): orientation-aligned pinch (thumb on flat, index on apex) + HAND CHOSEN BY ORIENTATION SCORE (both hands x 3 pinch dirs)"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --best-hand --out /data/evals/prism_besthand2.jsonl \
  --set '{"hand.closed": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], "hand.closed_triangle": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], "grasp.point_in_palm": [0.099, -0.084, 0.015], "grasp.depth_grasp_height": true, "perception.shape.enabled": true, "grasp.shape_grasp.enabled": true, "grasp.shape_profiles.triangle.enabled": false, "grasp.shape_profiles.cylinder.enabled": false, "grasp.shape_profiles.cube.enabled": false}' 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo PRISM6_DONE
