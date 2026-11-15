#!/bin/bash
# Train one shape end to end in the sim (task-3 scene: real lid layout, lid on the box floor):
#   1) expert demos (privileged expert + DART noise), right hand hands over when out of the left hand's reach
#   2) behaviour cloning   3) evaluation of the learned controller on NEW layouts
# usage (inside the container): train_shape.sh <shape> [n_demo=24] [n_eval=20]
S=${1:?shape}; N=${2:-24}; NE=${3:-20}; S0=${4:-0}
export TASK=Aalto-Insert-G129-Dex3-Joint AALTO_LID_LAYOUT=real AALTO_LID_HEIGHT=floor IL_LOG=1 IL_EVENTS=1
OUT=/data/il/$S; mkdir -p $OUT
cd /data/team/il_insert
echo "=== $S: $N expert demos ($(date +%T))"
aalto py /data/team/il_insert/run.py --mode expert --shape $S --noise 0.004 --seeds $(seq $S0 $((S0+N-1))) --time-limit 120 --out $OUT/demos 2>&1 \
  | grep --line-buffered -E "^###|EV |hybrid:|skip|Error"
echo "=== $S: training ($(date +%T))"
aalto py /data/team/il_insert/train_bc.py --demos $OUT/demos --out $OUT/bc.pt 2>&1 | grep --line-buffered -vE "Warn"
echo "=== $S: evaluation, $NE new layouts ($(date +%T))"
aalto py /data/team/il_insert/run.py --mode learned --model $OUT/bc.pt --shape $S --seeds $(seq 1000 $((1000+NE-1))) --time-limit 120 --out $OUT/eval 2>&1 \
  | grep --line-buffered -E "^###|hybrid:|Error"
echo "=== $S: done ($(date +%T))"
