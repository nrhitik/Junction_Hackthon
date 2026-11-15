#!/bin/bash
# Prism: aligned pinch (thumb on flat, index on apex) + MIDDLE finger SIDE as a wall (base joint bent, tip straight),
# hand shifted sideways towards the middle finger by 6 / 9 / 12 mm (point_in_palm z 0.015 -> 0.009 / 0.006 / 0.003).
cd /data/evals
WALL="[0.0, -0.02, 1.14, -1.35, -0.05, -1.12, -0.64]"
for Z in 0.009 0.006 0.003; do
  echo "=== WALL run: point_in_palm z = $Z"
  aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --out /data/evals/prism_wall_$Z.jsonl \
    --set "{\"hand.closed\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"hand.closed_triangle\": $WALL, \"grasp.point_in_palm\": [0.099, -0.084, $Z], \"grasp.depth_grasp_height\": true, \"perception.shape.enabled\": true, \"grasp.shape_grasp.enabled\": true, \"grasp.shape_profiles.triangle.enabled\": false, \"grasp.shape_profiles.cylinder.enabled\": false, \"grasp.shape_profiles.cube.enabled\": false}" 2>&1 \
    | grep --line-buffered -E "^LIFT|^  [a-z]|Traceback|Error"
done
echo WALL_DONE
