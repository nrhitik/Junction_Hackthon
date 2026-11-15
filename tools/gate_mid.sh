#!/bin/bash
cd /opt/aalto
export AALTO_OVERRIDES="{\"hand.closed\": [0.0, -0.02, 1.14, -1.12, -0.64, -1.12, -0.64]}" PORT=8000
aalto py /data/team/tools/grip_server.py > /data/evals/gate_mid_server.log 2>&1 &
SRV=$!
for _ in $(seq 60); do grep -q "loaded in" /data/evals/gate_mid_server.log && break; sleep 2; done
sleep 3
aalto eval --policy ws://127.0.0.1:8000 --task sort_fixed --trials 5 --seed 1000 --out /data/evals/gate_mid 2>&1
echo GATE_EXIT=$?
kill $SRV
