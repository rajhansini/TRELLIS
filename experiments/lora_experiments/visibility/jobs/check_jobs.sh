#!/bin/bash
# Compact status report for the visibility/rung8 jobs.
# Prints: what is queued, what finished, what FAILED and why, and what outputs exist.
# Designed to be cheap to run repeatedly from a watchdog loop.
#
#   bash experiments/lora_experiments/visibility/jobs/check_jobs.sh

ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
LOGDIR=${VIS}/logs
STATE=${VIS}/jobs/tracked_jobs.txt      # one job id per line, written at submit time

echo "############ $(date '+%Y-%m-%d %H:%M:%S') ############"

echo "--- QUEUE ---"
squeue -u rajhansini -o "%.10i %.14j %.8T %.10M %.20R" 2>&1 | grep -vE "test_f75|DependencyNeverSatisfied" || true

echo ""
echo "--- RECENT TERMINAL STATES (last 6h) ---"
sacct -u rajhansini -S "$(date -d '6 hours ago' '+%Y-%m-%dT%H:%M:%S')" \
      --format=JobID%12,JobName%14,State%12,ExitCode%8,Elapsed%10,End%20 -X 2>&1 \
  | grep -E "JobID|v4_360|rung8_isect|vis_probe|vis_orbit" || echo "  (none)"

echo ""
echo "--- FAILURES: first error line per failed job ---"
FAILED=$(sacct -u rajhansini -S "$(date -d '6 hours ago' '+%Y-%m-%dT%H:%M:%S')" \
         --format=JobID,JobName,State -X --noheader 2>/dev/null \
         | grep -E "v4_360|rung8_isect|vis_probe|vis_orbit" \
         | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED|NODE_FAIL" | awk '{print $1}')
if [ -z "$FAILED" ]; then
    echo "  none"
else
    for J in $FAILED; do
        F=$(ls ${LOGDIR}/*_${J}.log 2>/dev/null | head -1)
        echo "  job ${J}:"
        if [ -n "$F" ]; then
            grep -m3 -iE "error|traceback|assertionerror|illegal option|not found|cuda|no such file|killed" "$F" \
              | sed 's/^/      /' || echo "      (log has no obvious error line)"
            echo "      log: $F"
        else
            echo "      (no log file)"
        fi
    done
fi

echo ""
echo "--- OUTPUTS ---"
N_MP4=$(find ${VIS}/orbit_renders -name '*.mp4' 2>/dev/null | wc -l)
echo "  orbit mp4s        : ${N_MP4}"
[ "$N_MP4" -gt 0 ] && find ${VIS}/orbit_renders -name '*.mp4' -printf '      %p (%s bytes)\n' 2>/dev/null
echo "  visibility mask   : $([ -f ${VIS}/masks/visibility.npz ] && echo present || echo absent)"
R=$(ls -d ${VIS}/runs/rung8_* 2>/dev/null | head -1)
if [ -n "$R" ]; then
    NC=$(ls ${R}/lora_ckpts/lora_e*.pt 2>/dev/null | wc -l)
    echo "  rung8 run dir     : $(basename $R)   checkpoints=${NC}/30"
    if [ -f "${R}/logs/epoch_metrics.csv" ]; then
        echo "  last epoch row    :"
        tail -1 "${R}/logs/epoch_metrics.csv" | sed 's/^/      /'
    fi
    grep -h "SHIFT" ${LOGDIR}/rung8_*.log 2>/dev/null | tail -3 | sed 's/^/      /'
else
    echo "  rung8 run dir     : none yet"
fi
echo "###############################################"
