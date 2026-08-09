#!/bin/bash
# Per-head cross-attention analysis, FROZEN vs rung15.
#
# Every attention number so far was averaged over all 16 heads. Fuse3D reports
# only heads 1/5/13 are usable, so the head-mean may be 13 vague heads drowning
# 3 sharp ones. This decides that, and produces the per-head ranking the
# self-attention enhancement needs to know which voxels to target.
#
# Real script file, not sbatch --wrap: --wrap runs /bin/sh = dash, no pipefail.
#SBATCH --job-name=perhead
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/perhead_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/perhead_%j.log

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

${PY} -u "${VIS}/measure_perhead_attention.py" \
  --knots 0 12 24 \
  --run rung15v1_uniform_all_qkvo_r4_s6_2674ec70 \
  2>&1 | tee -a "${VIS}/logs/perhead_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
