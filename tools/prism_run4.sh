#!/bin/bash
cd /data/evals
echo "=== RUN F: TRIPOD: thumb on the apex, index + middle both pressing the opposite flat face (3 fingers), camera orientation"
aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --tripod --out /data/evals/prism_tripod.jsonl \
  --set '{"hand.closed": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64], "hand.closed_triangle": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64], "grasp.point_in_palm": [0.099, -0.084, 0.0], "grasp.depth_grasp_height": true, "perception.shape.enabled": true, "grasp.shape_grasp.enabled": true, "grasp.shape_profiles.triangle.enabled": false, "grasp.shape_profiles.cylinder.enabled": false, "grasp.shape_profiles.cube.enabled": false}' 2>&1 \
  | grep --line-buffered -E "^\{|^LIFT|^  [a-z]|Traceback|Error"
echo PRISM4_DONE
