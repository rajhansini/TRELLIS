#!/bin/bash
# Does the unseen surface read the video frame at all?
#
# Forms the cross-attention matrix A [7301 voxels x 1374 image tokens] that the
# fused kernel never materialises, recomputes visible/occluded at the FLOW
# resolution (7301, not the 467264 the existing visibility.npz labels), and
# compares the two groups' attention rows.
#
# Trains nothing. One forward pass per (frame, knot, block) cell = 27 cells.
#
# Real script file, not sbatch --wrap: --wrap runs /bin/sh = dash, no pipefail.
#SBATCH --job-name=xattn_corr
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/xattn_corr_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/xattn_corr_%j.log

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

${PY} -u "${VIS}/measure_xattn_correspondence.py" \
  --frames 1 75 150 --blocks 0 11 23 --knots 0 12 24 \
  2>&1 | tee -a "${VIS}/logs/xattn_corr_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
