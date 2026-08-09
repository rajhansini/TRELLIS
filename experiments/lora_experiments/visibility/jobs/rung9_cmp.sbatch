#!/bin/bash
# GT | frozen TRELLIS | rung9, 150 frames, one mp4.
# Real script file (not sbatch --wrap: that runs /bin/sh = dash, no pipefail).
#SBATCH --job-name=rung9_cmp
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung9_cmp_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung9_cmp_%j.log

set -eo pipefail

ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
LOGDIR=${VIS}/logs
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

mkdir -p "${LOGDIR}"
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cd "${ROOT}"
echo "=== attempt at $(date)  host=$(hostname)  job=${SLURM_JOB_ID} ==="
nvidia-smi -L

${PY} -u "${VIS}/render_rung9_comparison.py" \
  --rung9-run rung9_geom-color_r4_s6_e44d8696 --ckpt lora_best.pt \
  2>&1 | tee -a "${LOGDIR}/rung9_cmp_tee_${SLURM_JOB_ID}.log"

echo "=== Done at $(date) ==="
