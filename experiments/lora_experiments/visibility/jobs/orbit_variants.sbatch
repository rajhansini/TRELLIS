#!/bin/bash
#SBATCH --job-name=orb_var
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/orb_var_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/orb_var_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo "### rung11 — visibility mask"
${PY} -u "${VIS}/render_colonly_orbit.py" --sweep angle --frame 75 \
  --n-angles 120 --elevation 15 --fps 20 --no-gt \
  --run rung11_vismask_soft_r4_s6_7264f197 --panel-name "rung11 vismask" \
  --vismask "${VIS}/masks/visibility.npz" --mask-mode soft \
  --out-dir "${ROOT}/experiments/lora_experiments/runs/rung11_vismask_soft_r4_s6_7264f197/orbit"

echo "### rung13 — mesh alignment"
${PY} -u "${VIS}/render_colonly_orbit.py" --sweep angle --frame 75 \
  --n-angles 120 --elevation 15 --fps 20 --no-gt \
  --run rung13_aligned_r4_s6_c742c0b3 --panel-name "rung13 aligned" \
  --alignment "${VIS}/alignment/alignment.json" \
  --out-dir "${ROOT}/experiments/lora_experiments/runs/rung13_aligned_r4_s6_c742c0b3/orbit"

echo "=== Done $(date) ==="
