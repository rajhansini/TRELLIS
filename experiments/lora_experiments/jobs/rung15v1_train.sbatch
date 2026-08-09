#!/bin/bash
# rung14-v1 — LoRA on the SLaT-flow CROSS-ATTENTION (before the decoder), aligned.
#
# BUDGET. rung2 measured 792 s/epoch with 8 adapted blocks; 30 epochs took 6.6 h.
# This adapts 24, so expect >= 8 h against a 4 h walltime cap. The script
# checkpoints every epoch and resumes from the newest one, and --requeue +
# --open-mode=append makes a preempted job pick up where it stopped. Submit this
# 3x with afterany so the chain survives a timeout:
#
#   A=$(sbatch --parsable jobs/rung14v1_train.sbatch)
#   B=$(sbatch --parsable --dependency=afterany:$A jobs/rung14v1_train.sbatch)
#        sbatch            --dependency=afterany:$B jobs/rung14v1_train.sbatch
#
# afterany, not afterok, is correct ONLY because rung14v1_gates.sbatch has
# already proved the code runs. Chaining afterany on unproven code just repeats
# the same crash three times.
#
# Real script file, not sbatch --wrap: --wrap runs /bin/sh = dash, no pipefail.
#SBATCH --job-name=r15v1_train
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r15v1_train_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r15v1_train_%j.log

set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${LEX}/logs"
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

${PY} -u "${LEX}/rung15_v1_xattn_randt.py" \
  --rank 4 --epochs 30 --seed 6 --blocks all --targets qkvo --tsample uniform \
  2>&1 | tee -a "${LEX}/logs/r15v1_train_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
