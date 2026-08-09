#!/bin/bash
# rung17 gates. Loss-mask experiment; see rung17_e2_rembgmask_lpips.py header.
#SBATCH --job-name=r17_gates
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00

#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r17_gates_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r17_gates_%j.log
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
${PY} -u "${LEX}/rung17_e2_rembgmask_lpips.py" \
  --rank 4 --epochs 30 --seed 6 --blocks all --targets qkvo \
  --tsample uniform --train-mask rembg --gates-only \
  2>&1 | tee -a "${LEX}/logs/r17_gates_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
