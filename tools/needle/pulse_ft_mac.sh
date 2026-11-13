#!/bin/sh
# Pulse fine-tune runs on the Mac mini from an existing checkpoint (the second stage of pulse_run_mac.sh alone),
# one after the other. Experiments only.
#
#   screen -dmS pulseft sh ~/train/pulse_x/pulse_ft_mac.sh runs_pulse/p9_w24_l6_bal/last.pt beat_loops \
#       "p11_w24_l6_ft|" "p12_w24_l6_ftdw|--down-weight 3" -- --teacher-holdout ~/train/data/pulse_passfail/holdout_slow_quarter.txt
#
# Each run: "name|its own extra options". Options after -- go to every run. Logs: logs/<name>.log; a finished run
# (runs_pulse/<name>/done) is not repeated, a cut-off one resumes from its last check (every 5,000 steps).
set -u
INIT=${1:?checkpoint to start from}
LOOPS=${2:?folder of the synthetic loop tracks under ~/train/data}
shift 2
RUNS=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do RUNS="$RUNS
$1"; shift; done
[ $# -gt 0 ] && shift
ALL="$*"
cd "$(dirname "$0")" || exit 1
export PULSE_DATA="$HOME/train/data" PYTHONUNBUFFERED=1
PY="$HOME/train/.venv/bin/python"
mkdir -p logs
echo "$RUNS" | while IFS='|' read -r NAME OPTS; do
    [ -z "$NAME" ] && continue
    if [ ! -f "runs_pulse/$NAME/done" ]; then
        # shellcheck disable=SC2086
        caffeinate -ims "$PY" pulse_rave.py train --init "$INIT" --teacher "$PULSE_DATA/beat_teacher_ens" \
            --loops "$PULSE_DATA/$LOOPS" --loop-frac 0.2 --resume --steps 20000 --lr 5e-4 --name "$NAME" \
            $ALL $OPTS < /dev/null >> "logs/$NAME.log" 2>&1 || exit 1
        touch "runs_pulse/$NAME/done"
        echo "PULSE MAC DONE $(date +%Y-%m-%dT%H:%M:%S)" >> "logs/$NAME.log"
    fi
done
