#!/bin/bash
# Prism V-WEDGE grip: index + middle hang straight down (walls 41 mm apart), prism centred between them with its
# flat face to the thumb and apex forward; the thumb pushes the flat face -> both sloped faces wedge on the finger sides.
cd /data/evals
for TH in "0.0, 0.40, 0.76" "0.0, 0.52, 0.64"; do
  echo "=== WEDGE run: thumb ($TH)"
  aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --out "/data/evals/prism_wedge_${TH//[ ,]/_}.jsonl" \
    --set "{\"hand.closed\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"hand.closed_triangle\": [$TH, -1.35, -0.05, -1.35, -0.05], \"grasp.point_in_palm\": [0.099, -0.084, 0.0], \"perception.shape.triangle_pinch_shift\": 0.0, \"grasp.depth_grasp_height\": true, \"perception.shape.enabled\": true, \"grasp.shape_grasp.enabled\": true, \"grasp.shape_profiles.triangle.enabled\": false, \"grasp.shape_profiles.cylinder.enabled\": false, \"grasp.shape_profiles.cube.enabled\": false}" 2>&1 \
    | grep --line-buffered -E "^LIFT|^  [a-z]|Traceback|Error"
done
echo WEDGE_DONE
