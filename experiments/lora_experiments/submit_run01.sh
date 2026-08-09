#!/bin/bash
# Run 01 — Single-path LoRA on raw TRELLIS.
#
# Two-step submit:
#   BUILD_JOB : rebuild nvdiffrast for A40 (sm_86)
#   TRAIN_JOB : run01_single_path_lora.py, 20 epochs, depends on build
#
# Logs:
#   Build : slurm_logs/build_a40_<jobid>.log
#   Train : slurm_logs/run01_train_<jobid>.log
#   Python: run01_single_path_lora/train.log  (tee'd inside the script)

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/run01_single_path_lora.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

EPOCHS=${1:-20}

echo "=== Run 01 — Single-path LoRA (epochs=${EPOCHS}) ==="
echo ""

# ── Step 1: rebuild nvdiffrast for A40 (sm_86) ────────────────────────────────
echo "=== Step 1: nvdiffrast build for A40 ==="

BUILD_JOB=$(sbatch \
    --job-name="build_a40" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGDIR}/build_a40_%j.log" \
    --error="${LOGDIR}/build_a40_%j.log" \
    --wrap="
set -e
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
bash ${BUILD_SCRIPT}
" | awk '{print $NF}')

echo "  Build job : ${BUILD_JOB}"
echo "  Build log : ${LOGDIR}/build_a40_${BUILD_JOB}.log"
echo ""

# ── Step 2: training ──────────────────────────────────────────────────────────
echo "=== Step 2: Run 01 training (depends on build ${BUILD_JOB}) ==="

TRAIN_JOB=$(sbatch \
    --job-name="run01_lora" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --requeue \
    --dependency=afterok:${BUILD_JOB} \
    --output="${LOGDIR}/run01_train_%j.log" \
    --error="${LOGDIR}/run01_train_%j.log" \
    --wrap="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

echo '=== GPU info ==='
nvidia-smi -L
echo ''
echo '=== Run 01: single-path LoRA ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} --epochs ${EPOCHS}

echo '=== Run 01 complete ==='
" | awk '{print $NF}')

echo "  Train job : ${TRAIN_JOB}  (waits on build ${BUILD_JOB})"
echo "  Slurm log : ${LOGDIR}/run01_train_${TRAIN_JOB}.log"
echo "  Python log: /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/run01_single_path_lora/train.log"
echo ""
echo "=== Submitted (build ${BUILD_JOB} → train ${TRAIN_JOB}) ==="
echo ""
echo "Monitor:      squeue -u rajhansini"
echo "Tail build:   tail -f ${LOGDIR}/build_a40_${BUILD_JOB}.log"
echo "Tail train:   tail -f ${LOGDIR}/run01_train_${TRAIN_JOB}.log"
echo "Python log:   tail -f /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/run01_single_path_lora/train.log"
echo ""
echo "Results after run:"
echo "  loss_history.json"
echo "  lora_ckpts/lora_best.pt"
echo "  diag_renders/"
