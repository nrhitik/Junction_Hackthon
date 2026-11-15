#!/bin/bash
# Wait for the wedge runs, then: wedge grip DEEP (prism grip at 30% height, fingertips may come to 5 mm of the counter)
cd /data/evals
while ! grep -q WEDGE_DONE /data/evals/prism_wedge.out 2>/dev/null; do sleep 10; done
docker_cp_note=1
for TH in "0.0, 0.52, 0.64" "0.0, 0.40, 0.76"; do
  echo "=== DEEP WEDGE run: thumb ($TH), height_frac 0.3, fingertip clearance 5 mm"
  aalto py /data/team/tools/lift_test.py --seed 7 --shapes triangle --copies 2 --aligned --out "/data/evals/prism_deep_${TH//[ ,]/_}.jsonl" \
    --set "{\"hand.closed\": [0.0, -0.02, 1.14, -0.1, -0.05, -1.12, -0.64], \"hand.closed_triangle\": [$TH, -1.35, -0.05, -1.35, -0.05], \"grasp.point_in_palm\": [0.099, -0.084, 0.0], \"perception.shape.triangle_pinch_shift\": 0.0, \"grasp.depth_grasp_height\": true, \"grasp.shape_height_frac\": {\"triangle\": 0.3}, \"grasp.shape_fingertip_clearance\": {\"triangle\": 0.005}, \"perception.shape.enabled\": true, \"grasp.shape_grasp.enabled\": true, \"grasp.shape_profiles.triangle.enabled\": false, \"grasp.shape_profiles.cylinder.enabled\": false, \"grasp.shape_profiles.cube.enabled\": false}" 2>&1 \
    | grep --line-buffered -E "^LIFT|^  [a-z]|Traceback|Error"
done
echo DEEP_DONE
