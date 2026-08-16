#!/bin/bash
# rung21 CALIBRATION: measure KL vs loss magnitude. No optimiser step, no ckpt.
#SBATCH --job-name=r21_w253e03
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r21_w253e03_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r21_w253e03_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
${PY} -u "${LEX}/rung21_kl.py" \
  --rank 4 --epochs 30 --seed 6 --blocks all --targets kv \
  --tsample uniform --loss-region intersection \
  --kl --kl-blocks all --w-kl 2.53e-03 \
  2>&1 | tee -a "${LEX}/logs/r21_w253e03_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
