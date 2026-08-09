#!/bin/bash
#SBATCH --job-name=lvar
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/lvar_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/lvar_%j.log
set -eo pipefail
export SPCONV_ALGO=native ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
  experiments/lora_experiments/visibility/measure_lora_mesh_variation.py \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
