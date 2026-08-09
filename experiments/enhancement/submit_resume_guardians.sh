#!/bin/bash
# Guardian jobs for all 4 pinned-alpha runs.
#
# Each job:
#   1. Checks if epoch 20 checkpoint already exists (interactive run finished) -> exits
#   2. Otherwise runs step07d, which auto-resumes from the latest checkpoint
#   3. --requeue: if the node dies, SLURM requeues automatically and we resume again
#
# Submit ONCE now. If your interactive run finishes first, the guardian exits fast.
# If you lose the interactive node, the guardian picks up from the last checkpoint.

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07d_lora_pinned_alpha.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

MODE=v2_C
EPOCHS=20
RANK=4
LOSS_SCALE=4096

# Format: "ALPHA:LR:RESULTS_SUFFIX"
RUNS=(
    "0.1:1e-4:alpha0p10_seed6"
    "0.25:1e-4:alpha0p25_seed6"
    "0.5:1e-4:alpha0p50_seed6"
    "0.25:5e-4:alpha0p25_lr5e-04_seed6"
)

echo "=== Submitting resume guardians: mode=${MODE}, epochs=${EPOCHS} ==="
echo "    Runs: ${#RUNS[@]} (alphas 0.1/0.25/0.5 at lr=1e-4 + alpha=0.25 at lr=5e-4)"
echo ""

for RUN in "${RUNS[@]}"; do
    ALPHA=$(echo $RUN | cut -d: -f1)
    LR=$(echo $RUN | cut -d: -f2)
    SUFFIX=$(echo $RUN | cut -d: -f3)

    RESULTS_DIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_d_${SUFFIX}
    DONE_CKPT=${RESULTS_DIR}/lora_ckpts/lora_e${EPOCHS}.pt
    JOB_TAG=$(echo $SUFFIX | sed 's/_seed6//' | sed 's/alpha/a/' | sed 's/lr/l/')

    WRAP="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

echo '=== Guardian: alpha=${ALPHA} lr=${LR} ==='
nvidia-smi -L

if [ -f '${DONE_CKPT}' ]; then
    echo '[GUARDIAN] epoch ${EPOCHS} checkpoint found — already done, exiting.'
    exit 0
fi

echo '[GUARDIAN] resuming from latest checkpoint (or fresh if none)'
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --alpha ${ALPHA} \
    --epochs ${EPOCHS} \
    --rank ${RANK} \
    --lr ${LR} \
    --loss_scale ${LOSS_SCALE}

echo '[GUARDIAN] step07d exited cleanly.'
"

    JOB=$(sbatch \
        --job-name="grd_${JOB_TAG}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --requeue \
        --output="${LOGDIR}/guardian_${SUFFIX}_%j.log" \
        --error="${LOGDIR}/guardian_${SUFFIX}_%j.log" \
        --wrap="$WRAP" | awk '{print $NF}')

    echo "  alpha=${ALPHA} lr=${LR}  job=${JOB}"
    echo "  results: ${RESULTS_DIR}"
    echo "  slurm log: ${LOGDIR}/guardian_${SUFFIX}_${JOB}.log"
    echo ""
done

echo "=== All 4 guardians submitted ==="
echo ""
echo "Monitor:  squeue -u rajhansini"
echo "Logs:     ls ${LOGDIR}/"
