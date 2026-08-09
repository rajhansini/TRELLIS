#!/bin/bash
#SBATCH --job-name=evalmatch
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/evalmatch_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/evalmatch_%j.log
set -eo pipefail
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
${PY} -u experiments/lora_experiments/visibility/eval_match.py \
  --run rung19_intersection_uniform_all_qkvo_r4_s6_276b5db7 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
