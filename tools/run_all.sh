#!/bin/bash
# Whole stack, one 120 s episode per task (seed 5).
cd /data/evals
for T in sort_fixed sort_language; do
  echo "################ TASK $T"
  bash /data/team/tools/fb_ab.sh stack_$T "5" "{}" "--task $T"
done
echo "################ restarting the sim with the Insert scene (lid)"
aalto stop >/dev/null 2>&1; sleep 3
TASK=Aalto-Insert-G129-Dex3-Joint aalto sim 2>&1 | tail -1
echo "################ TASK insert_shapes"
TASK=Aalto-Insert-G129-Dex3-Joint bash /data/team/tools/fb_ab.sh stack_insert_shapes "5" "{}" "--task insert_shapes"
echo "################ back to the sort scene"
aalto stop >/dev/null 2>&1; sleep 3; aalto sim 2>&1 | tail -1
echo ALL_DONE
