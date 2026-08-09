#!/bin/bash
# rung13 — camera orbits 360 deg WHILE the frame advances 1..150.
# Both the dynamic texture and the viewpoint change together.
#SBATCH --job-name=r13_orb_both
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/r13_orb_both_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/r13_orb_both_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
${PY} -u "${VIS}/render_colonly_orbit.py" --sweep both \
  --n-angles 150 --n-frames 150 --elevation 15 --fps 15 --no-gt \
  --run rung13_aligned_r4_s6_c742c0b3 --panel-name "rung13 ALIGNED (ours)" \
  --alignment "${VIS}/alignment/alignment.json" \
  --out-dir "${ROOT}/experiments/lora_experiments/runs/rung13_aligned_r4_s6_c742c0b3/orbit_both" \
  2>&1 | tee -a "${VIS}/logs/r13_orb_both_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
