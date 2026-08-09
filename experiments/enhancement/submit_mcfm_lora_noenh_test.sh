#!/bin/bash
# Smoke test — 1 frame, 1 epoch, all 4 modes.
# Run this first, read logs, then submit submit_mcfm_lora_noenh.sh for full training.
#
# Logs (read these):
#   Python Tee : results_mcfm_{mode}_noenh_lora_seed6/train_mode{mode}_noenh_epochs1.log
#   Slurm out  : slurm_logs/lora_noenh_test_{mode}_{jobid}.log

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07b_mcfm_lora_noenh.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

EPOCHS=1
LR=1e-4
RANK=4
LOSS_SCALE=4096
FRAME_STRIDE=150   # only frame 1 — fast smoke test

echo "=== Submitting SMOKE TEST: 1 frame, 1 epoch, all 4 modes ==="

for MODE in v2_C v2_D v3_C v3_D; do
    JOB=$(sbatch \
        --job-name="noenh_test_${MODE}" \
        --partition=general \
        --gres=gpu:L40S:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=01:00:00 \
        --output="${LOGDIR}/lora_noenh_test_${MODE}_%j.log" \
        --error="${LOGDIR}/lora_noenh_test_${MODE}_%j.log" \
        --wrap="
set -e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

echo '=== GPU info ==='
nvidia-smi -L
echo ''
echo '=== SMOKE TEST: mode=${MODE} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --rank ${RANK} \
    --loss_scale ${LOSS_SCALE} \
    --frame_stride ${FRAME_STRIDE}

echo '=== SMOKE TEST DONE: mode=${MODE} ==='
")
    echo "  ${MODE}: ${JOB}"
done

echo ""
echo "=== Where to read logs ==="
for MODE in v2_C v2_D v3_C v3_D; do
    echo "  Python Tee : /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_noenh_lora_seed6/train_mode${MODE}_noenh_epochs1.log"
done
echo "  Slurm out  : ${LOGDIR}/lora_noenh_test_{mode}_{jobid}.log"
echo ""
echo "Check logs. If GATE 1 PASSED, GATE 2 PASSED, W1-W4 all good => run submit_mcfm_lora_noenh.sh"
