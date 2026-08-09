#!/bin/bash
# Step 07c: real video GT + alpha² regularization training — all 4 modes
#
# Fixes:
#   FIX 1: GT = real Kling video frames (NOT MCFM renders)
#   FIX 2: loss += ALPHA_REG * alpha² — keeps alpha small
#
# Evidence at every step:
#   Python log : results_mcfm_{mode}_realgt_lora_seed6/train_mode{mode}_realgt_epochs50.log
#   Diagnostics: results_mcfm_{mode}_realgt_lora_seed6/diag_renders/e{N:03d}_f0075.png
#   Loss JSON  : results_mcfm_{mode}_realgt_lora_seed6/loss_history.json
#   Best ckpt  : results_mcfm_{mode}_realgt_lora_seed6/lora_ckpts/lora_best.pt
#   Slurm log  : slurm_logs/lora_realgt_{mode}_r{1..6}_{jobid}.log
#
# 6 chained 4h afterany jobs per mode (24h coverage, auto-resume)

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07c_mcfm_lora_realgt.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

EPOCHS=50
LR=1e-4
RANK=4
LOSS_SCALE=4096
ALPHA_REG=0.01   # L2 penalty on alpha — tune if alpha still drifts up

echo "=== Submitting step07c: real GT + alpha² reg, ${EPOCHS} epochs, all 4 modes ==="
echo "    ALPHA_REG=${ALPHA_REG}  LR=${LR}  RANK=${RANK}  LOSS_SCALE=${LOSS_SCALE}"
echo ""

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
echo '=== step07c: mode=${MODE} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --rank ${RANK} \
    --loss_scale ${LOSS_SCALE} \
    --alpha_reg ${ALPHA_REG}

echo '=== DONE: mode=${MODE} ==='
"

    PREV=""
    for R in 1 2 3 4 5 6; do
        if [ -z "$PREV" ]; then
            DEP=""
        else
            DEP="--dependency=afterany:${PREV}"
        fi

        JOB=$(sbatch \
            --job-name="rg_${MODE}_r${R}" \
            --partition=general \
            --gres=gpu:L40S:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=04:00:00 \
            $DEP \
            --output="${LOGDIR}/lora_realgt_${MODE}_r${R}_%j.log" \
            --error="${LOGDIR}/lora_realgt_${MODE}_r${R}_%j.log" \
            --wrap="$WRAP" | awk '{print $NF}')

        echo "  ${MODE} r${R}: job=${JOB}  dep=${PREV:-none}"
        PREV=$JOB
    done
    echo ""
done

echo "=== Smoke test commands (run first to verify before full training) ==="
for MODE in v2_C v2_D v3_C v3_D; do
    echo "  ${PY} ${SCRIPT} --mode ${MODE} --smoke --alpha_reg ${ALPHA_REG}"
done

echo ""
echo "=== Log locations ==="
for MODE in v2_C v2_D v3_C v3_D; do
    echo "  ${PY} log  : /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_realgt_lora_seed6/train_mode${MODE}_realgt_epochs50.log"
    echo "  diag      : /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_realgt_lora_seed6/diag_renders/"
    echo "  best ckpt : /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_realgt_lora_seed6/lora_ckpts/lora_best.pt"
done
echo "  slurm logs: ${LOGDIR}/lora_realgt_{mode}_r{1..6}_{jobid}.log"
