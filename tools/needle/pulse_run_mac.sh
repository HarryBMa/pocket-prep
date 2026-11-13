#!/bin/sh
# Pulse training on the Mac mini (see pulse_mac.py): the p3 recipe (60,000 steps from scratch, lr 2e-3), then
# the p4 recipe (20,000 more steps from that checkpoint, lr 5e-4), as in pulse_q3b.ps1 and the p4 run.
#
#   screen -dmS pulse sh ~/train/pulse/pulse_run_mac.sh p5_w24_l6 p6_w24_l6 beat_loops_1t
#   screen -dmS pulse sh ~/train/pulse_x/pulse_run_mac.sh p9 p10 beat_loops --teacher-balance 0.5    (an experiment)
#
# Logs: ~/train/pulse/logs/<run>.log. A run that was cut off carries on from its last check (every 5,000
# steps) when the same command is given again; a finished run (runs_pulse/<run>/done) is not repeated.
set -u
A=${1:?name of the from-scratch run}
B=${2:?name of the fine-tune run}
LOOPS=${3:?folder of the synthetic loop tracks under ~/train/data}
shift 3                                         # anything after the three names goes to both runs (experiments)
cd "$(dirname "$0")" || exit 1
export PULSE_DATA="$HOME/train/data" PYTHONUNBUFFERED=1
PY="$HOME/train/.venv/bin/python"
COMMON="--teacher $PULSE_DATA/beat_teacher_ens --loops $PULSE_DATA/$LOOPS --loop-frac 0.2 --resume $*"
mkdir -p logs
if [ ! -f "runs_pulse/$A/done" ]; then
    caffeinate -ims "$PY" pulse_rave.py train --init none --width 24 --levels 6 --seed 1 $COMMON \
        --steps 60000 --lr 2e-3 --name "$A" >> "logs/$A.log" 2>&1 || exit 1
    touch "runs_pulse/$A/done"
fi
if [ ! -f "runs_pulse/$B/done" ]; then
    caffeinate -ims "$PY" pulse_rave.py train --init "runs_pulse/$A/last.pt" $COMMON \
        --steps 20000 --lr 5e-4 --name "$B" >> "logs/$B.log" 2>&1 || exit 1
    touch "runs_pulse/$B/done"
fi
echo "PULSE MAC DONE $(date +%Y-%m-%dT%H:%M:%S)" >> "logs/$B.log"
