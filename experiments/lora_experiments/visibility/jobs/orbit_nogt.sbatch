#!/bin/bash
#SBATCH --job-name=orbit_nogt
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/orbit_nogt_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/orbit_nogt_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
${PY} -u "${VIS}/render_colonly_orbit.py" --sweep angle --frame 75 \
  --n-angles 120 --elevation 15 --fps 20 --no-gt \
  2>&1 | tee -a "${VIS}/logs/orbit_nogt_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
