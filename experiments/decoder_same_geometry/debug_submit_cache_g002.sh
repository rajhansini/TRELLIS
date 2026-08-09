#!/bin/bash
# Debug: pin to g002 (L40S) — the exact node all rung5 training ran on.
# g002 is confirmed to give N_vox=7301.
# Start with frame 75 only (~5 min) to verify before running all 150.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

MODE="${1:-frame75}"   # "frame75" (default) or "all"

if [[ "$MODE" == "all" ]]; then
    FRAME_ARG=""
    TIMELIMIT="01:30:00"
    JOBNAME="slat_cache_g002_all"
else
    FRAME_ARG="--frame 75"
    TIMELIMIT="00:20:00"
    JOBNAME="slat_cache_g002_f75"
fi

echo "Mode: $MODE  frame_arg='$FRAME_ARG'  time=$TIMELIMIT"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOG_DIR}/${JOBNAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOBNAME}_%j.err
#SBATCH --nodelist=g002
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIMELIMIT}
#SBATCH --partition=general

echo "=== node: \$(hostname)  gpu: \$(nvidia-smi -L | head -1) ==="

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline

\$PY ${SCRIPT_DIR}/save_slat_cache.py ${FRAME_ARG}
echo "exit code: \$?"
EOF
