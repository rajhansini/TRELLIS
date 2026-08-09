#!/bin/bash
# The two attention tables for the paper: FROZEN vs rung15-v1, all 150 frames.
#
# Regenerates A = softmax(q @ k.T) from (weights, seed, frame) -- the identical
# computation the training run performed. Attention is never stored by anything,
# so this is the only way to obtain it, and it is exact rather than approximate.
#
# One 25-step ODE pass per frame per model captures all 9 (knot, block) cells.
# 150 frames x 2 models = 300 passes.
#
# Real script file, not sbatch --wrap: --wrap runs /bin/sh = dash, no pipefail.
#SBATCH --job-name=logits
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/delta_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/delta_%j.log

set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${VIS}/logs"
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L

${PY} -u "${VIS}/measure_delta.py" \
  --frame 75 --tau 4.0 \
  --run rung16e1_rembg_uniform_all_qkvo_r4_s6_0f8620e0 \
  2>&1 | tee -a "${VIS}/logs/delta_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
