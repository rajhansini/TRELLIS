#!/bin/bash
# rung13, fixed TRAINING camera, all 150 frames — the DYNAMIC/temporal video.
# Camera never moves; the only thing changing is the lava evolving frame to frame.
#SBATCH --job-name=r13_temporal
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/r13_temporal_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/r13_temporal_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
${PY} -u "${VIS}/render_colonly_comparison.py" \
  --run rung13_aligned_r4_s6_c742c0b3 --ckpt lora_best.pt --n-frames 150 --fps 15 \
  --alignment "${VIS}/alignment/alignment.json" \
  --panel-name "rung13 ALIGNED (ours)" \
  --out-dir "${ROOT}/experiments/lora_experiments/runs/rung13_aligned_r4_s6_c742c0b3/temporal" \
  2>&1 | tee -a "${VIS}/logs/r13_temporal_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
