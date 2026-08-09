#!/bin/bash
#SBATCH --job-name=evalm20
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/evalm20_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/evalm20_%j.log
set -eo pipefail
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
${PY} -u experiments/lora_experiments/visibility/eval_match.py --out /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/eval_match_r20 \
  --run rung20_intersection_uniform_all_kv_r4_s6_c7a82c89 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
