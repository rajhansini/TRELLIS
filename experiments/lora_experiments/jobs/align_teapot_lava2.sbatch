#!/bin/bash
#SBATCH --job-name=align_teapot_lava2
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/align_teapot_lava2_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/align_teapot_lava2_%j.log
set -eo pipefail
export SPCONV_ALGO=native ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
V=experiments/lora_experiments/visibility
/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
  ${V}/solve_align_obj.py \
  --slat-npz ${V}/alignment_teapot_lava2/slat_cache.npz \
  --frames-dir /net/projects/ranalab/rajhansini/TRELLIS.2/data/teapot_lava2/frames_from_video \
  --out-dir ${V}/alignment_teapot_lava2 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
