#!/bin/bash
# Rung 13 — rung5_colonly + the solved 3D mesh alignment. One variable.
#
# Depends on visibility/alignment/alignment.json (job 2148935) — the script
# raises if it is missing rather than silently training unaligned.
#
# Real script file (sbatch --wrap runs /bin/sh = dash, no pipefail).
# --requeue + --open-mode=append: resumes from its latest per-epoch checkpoint.
#SBATCH --job-name=rung13_align
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung13_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung13_%j.log

set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
VIS=${LEX}/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${VIS}/logs"
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi -L
echo "--- alignment being used ---"
cat "${VIS}/alignment/alignment.json" | head -20

${PY} -u "${LEX}/rung13_aligned_colonly_lora.py" \
  --rank 4 --epochs 30 --seed 6 \
  2>&1 | tee -a "${VIS}/logs/rung13_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
