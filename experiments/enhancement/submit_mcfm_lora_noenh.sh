#!/bin/bash
# Self-consistency training — 50 epochs, all 150 frames, all 4 modes.
# GT = MCFM rendered frames (results_mcfm_{mode}_seed6_fixednoise/beta0p0/)
#
# AUTO-RESUME: 6 chained 4h jobs per mode (6×4h = 24h coverage).
#   - afterany: each job runs regardless of whether previous succeeded or failed
#   - step07b auto-resumes from latest checkpoint on startup
#   - step07b exits cleanly if already at epoch 50 (no wasted GPU time)
#   - L40S nodes only (nvdiffrast compiled and working)
#
# Logs:
#   Python Tee : results_mcfm_{mode}_selfcons_lora_seed6/train_mode{mode}_noenh_epochs50.log
#   Slurm      : slurm_logs/lora_selfcons_{mode}_r{1..6}_{jobid}.log

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07b_mcfm_lora_noenh.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

EPOCHS=50
LR=1e-4
RANK=4
LOSS_SCALE=4096

echo "=== Submitting SELF-CONSISTENCY TRAINING: ${EPOCHS} epochs, 150 frames, all 4 modes (6x4h afterany, L40S only) ==="

for MODE in v2_C v2_D v3_C v3_D; do

    WRAP="
set -e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

echo '=== GPU info ==='
nvidia-smi -L
echo ''
echo '=== LoRA selfcons: mode=${MODE} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --rank ${RANK} \
    --loss_scale ${LOSS_SCALE}

echo '=== DONE: mode=${MODE} ==='
"

    # Chain 6 jobs with afterany — runs regardless of previous exit status
    PREV=""
    for R in 1 2 3 4 5 6; do
        if [ -z "$PREV" ]; then
            DEP=""
        else
            DEP="--dependency=afterany:${PREV}"
        fi

        JOB=$(sbatch \
            --job-name="sc_${MODE}_r${R}" \
            --partition=general \
            --gres=gpu:L40S:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=04:00:00 \
            $DEP \
            --output="${LOGDIR}/lora_selfcons_${MODE}_r${R}_%j.log" \
            --error="${LOGDIR}/lora_selfcons_${MODE}_r${R}_%j.log" \
            --wrap="$WRAP" | awk '{print $NF}')

        echo "  ${MODE} r${R}: job=${JOB}  dep=${PREV:-none}"
        PREV=$JOB
    done
    echo ""
done

echo "=== Where to read logs ==="
for MODE in v2_C v2_D v3_C v3_D; do
    echo "  Python Tee : /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_selfcons_lora_seed6/train_mode${MODE}_noenh_epochs50.log"
done
echo "  Slurm      : ${LOGDIR}/lora_selfcons_{mode}_r{1..6}_{jobid}.log"
