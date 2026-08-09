#!/bin/bash
# Submit LoRA training for all 4 MCFM modes: v2_C, v2_D, v3_C, v3_D
# Usage: bash submit_mcfm_lora.sh
# Each mode gets its own job. Auto-resumes from checkpoint if resubmitted.

set -e

BETA=6.0
EPOCHS=50
LR=1e-4
RANK=4
LOSS_SCALE=4096

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07_mcfm_lora_train.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs

echo "=== Submitting MCFM LoRA jobs: beta=${BETA} epochs=${EPOCHS} ==="

for MODE in v2_C v2_D v3_C v3_D; do
    JOB=$(sbatch \
        --job-name="lora_${MODE}" \
        --partition=general \
        --gres=gpu:L40S:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        --output="${LOGDIR}/lora_${MODE}_%j.log" \
        --error="${LOGDIR}/lora_${MODE}_%j.log" \
        --wrap="
set -e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

echo '=== GPU info ==='
nvidia-smi -L

echo '=== Starting LoRA: mode=${MODE} beta=${BETA} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --beta ${BETA} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --rank ${RANK} \
    --loss_scale ${LOSS_SCALE}

echo '=== Done: mode=${MODE} ==='
")
    echo "  ${MODE}: ${JOB}"
done

echo ""
echo "Logs:    ${LOGDIR}/lora_<mode>_<jobid>.log"
echo "Results: .../enhancement/results_mcfm_<mode>_lora_seed6/"
echo "Ckpts:   .../enhancement/results_mcfm_<mode>_lora_seed6/lora_ckpts/"
