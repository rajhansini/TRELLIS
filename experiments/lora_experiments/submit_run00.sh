#!/bin/bash
# Run 00 — Baseline: raw TRELLIS per frame, no MCFM, no LoRA.
#
# Two-step submit:
#   BUILD_JOB : rebuild nvdiffrast for A40 (sm_86)
#   EVAL_JOB  : run00_baseline.py, depends on build
#
# Output:  lora_experiments/run00_baseline/
#            renders/frame_XXXX.png
#            metrics.json
#            metrics_summary.txt
#
# Logs:
#   Build   : slurm_logs/build_a40_<jobid>.log
#   Eval    : slurm_logs/run00_baseline_<jobid>.log

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/run00_baseline.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

# ── Step 1: rebuild nvdiffrast for A40 (sm_86) ────────────────────────────────
echo "=== Step 1: Submit nvdiffrast rebuild for A40 (sm_86) ==="

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

# ── Step 2: baseline eval ─────────────────────────────────────────────────────
echo "=== Step 2: Submit run00_baseline (depends on build ${BUILD_JOB}) ==="

EVAL_JOB=$(sbatch \
    --job-name="run00_base" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --dependency=afterok:${BUILD_JOB} \
    --output="${LOGDIR}/run00_baseline_%j.log" \
    --error="${LOGDIR}/run00_baseline_%j.log" \
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
echo '=== Run 00: Baseline, no MCFM, no LoRA ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT}

echo '=== Run 00 complete ==='
" | awk '{print $NF}')

echo "  Eval job  : ${EVAL_JOB}  (waits on build job ${BUILD_JOB})"
echo "  Eval log  : ${LOGDIR}/run00_baseline_${EVAL_JOB}.log"
echo ""
echo "=== Both jobs submitted ==="
echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo ""
echo "Tail logs:"
echo "  tail -f ${LOGDIR}/build_a40_${BUILD_JOB}.log"
echo "  tail -f ${LOGDIR}/run00_baseline_${EVAL_JOB}.log"
echo ""
echo "Results (after eval completes):"
echo "  cat /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/run00_baseline/metrics_summary.txt"
