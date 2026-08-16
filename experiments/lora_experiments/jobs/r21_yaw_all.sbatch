#!/bin/bash
# Export frame-1 mesh for each completed v1 KL arm, then yaw-sweep it.
# The verdict number: bright fraction at yaw 270 vs yaw 0.
#SBATCH --job-name=r21_yaw
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r21_yaw_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r21_yaw_%j.log
set -eo pipefail
export SPCONV_ALGO=native ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
while IFS=$'\t' read -r W R; do
  echo ""; echo "############ w_kl=$W  run=$R ############"
  O=render/out/r21_${W}_f0001
  ${PY} -u render/export_xattn_mesh.py --run "$R" --frame 1 --no-glb --out-dir "$O" \
    2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|it/s\]" | tail -8
  ${PY} -u experiments/lora_experiments/visibility/yaw_bright_sweep.py \
    --mesh "${O}/result_f0001.ply" --frozen "${O}/frozen_f0001.ply" \
    2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn" | tail -20
done < /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs/r21_done.tsv
echo "=== Done $(date) ==="
