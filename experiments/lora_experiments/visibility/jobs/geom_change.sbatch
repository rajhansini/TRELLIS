#!/bin/bash
# Does the FROZEN TRELLIS decoder already give time-varying geometry?
# Decodes the frozen mesh across the sequence and compares 3D surface motion and
# silhouette against frame 1, alongside the GT video's own silhouette change.
#
# Real script file, not sbatch --wrap: --wrap runs under /bin/sh (dash here),
# which has no `set -o pipefail`.
#
# Submit:
#   sbatch experiments/lora_experiments/visibility/jobs/geom_change.sbatch
#SBATCH --job-name=geom_change
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/geom_change_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/geom_change_%j.log

set -eo pipefail

ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
LOGDIR=${VIS}/logs
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cd "${ROOT}"

echo "=================================================================="
echo "attempt at $(date)  host=$(hostname)  job=${SLURM_JOB_ID}"
nvidia-smi -L
echo "=================================================================="

${PY} -u "${VIS}/measure_geometry_change.py" \
  --run-id c85c888f --stride 10 --n-sample 20000 \
  2>&1 | tee -a "${LOGDIR}/geom_change_tee_${SLURM_JOB_ID}.log"

echo "=== Done at $(date) ==="
