#!/bin/bash
# rung14-v1 GATES ONLY — ~10 min. Run this BEFORE the 4 h training slot.
#
# Proves, on a real GPU, that:
#   GATE 0          param count matches the real Linear shapes
#   GATE-align      the alignment transform reaches the renderer
#   GATE-identity   at B=0 the patched cross-attn == frozen cross-attn
#   GATE-grad       all B matrices get non-zero gradient, and prints PEAK GPU
#
# The peak-GPU number is the point. rung2 ran 8 adapted blocks at 11.7 GB;
# this runs 24 with the frozen projections kept in the graph, so the footprint
# is an open question and an OOM three hours into training is the expensive
# way to answer it.
#
# Real script file, not sbatch --wrap: --wrap runs /bin/sh = dash, no pipefail.
#SBATCH --job-name=r14v1_gate
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r14v1_gates_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r14v1_gates_%j.log

set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${LEX}/logs"
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

${PY} -u "${LEX}/rung14_v1_crossattn_lora.py" \
  --rank 4 --seed 6 --blocks all --targets qkvo --gates-only \
  2>&1 | tee -a "${LEX}/logs/r14v1_gates_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
