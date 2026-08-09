#!/bin/bash
# Rung 8 — intersection-loss colour LoRA (white-cast fix), 30 epochs.
#
# A REAL script file, not sbatch --wrap. --wrap executes under /bin/sh, which is
# dash on this cluster, and dash has no `set -o pipefail` — jobs 2143676/2143677
# died in 0 seconds with "Illegal option -o pipefail".
#
# --requeue + --open-mode=append: on preemption slurm restarts this same script,
# rung8_intersection_lora.py finds its latest per-epoch checkpoint and resumes,
# and every log appends rather than truncating.
#
# Submit:
#   sbatch experiments/lora_experiments/visibility/jobs/rung8.sbatch
#SBATCH --job-name=rung8_isect
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung8_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung8_%j.log

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
echo "restart count = ${SLURM_RESTART_COUNT:-0}"
nvidia-smi -L
echo "=================================================================="

${PY} -u "${VIS}/rung8_intersection_lora.py" \
  --rank 4 --epochs 30 --seed 6 \
  --lambda-reg 0.0 --v4-run-id c85c888f --loss-norm v4compat \
  2>&1 | tee -a "${LOGDIR}/rung8_tee_${SLURM_JOB_ID}.log"

echo "=== Done at $(date) ==="
