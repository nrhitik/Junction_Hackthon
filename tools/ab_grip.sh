#!/bin/bash
# A/B the Dex3 closed posture (demo-derived squeeze depth). Seeds 0-4 per variant.
cd /data/evals
declare -A V=(
  [base]="{}"
  [mid]="{\"hand.closed\": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64]}"
  [deep]="{\"hand.closed\": [0.0, 0.06, 1.08, -1.24, -0.68, -1.24, -0.68]}"
)
for k in base mid deep; do
  echo "=== $k ${V[$k]}"
  rm -f ab_$k.jsonl
  aalto py /data/team/tools/diag_attempts.py --seeds 0 1 2 3 4 --out /data/evals/ab_$k.jsonl --set "${V[$k]}" 2>&1 | grep -E "^###|Error|Traceback"
done
echo AB_DONE
