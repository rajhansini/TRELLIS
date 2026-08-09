#!/bin/bash
#SBATCH --job-name=yaw20
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:20:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/yaw20_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/yaw20_%j.log
set -eo pipefail
export SPCONV_ALGO=native ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
  experiments/lora_experiments/visibility/yaw_bright_sweep.py --mesh /net/projects/ranalab/rajhansini/TRELLIS/render/out/rung20_i_f0001/result_f0001.ply --frozen /net/projects/ranalab/rajhansini/TRELLIS/render/out/rung20_i_f0001/frozen_f0001.ply \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
