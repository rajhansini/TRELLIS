#!/bin/bash
#SBATCH --job-name=export150
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamesh_writeup/figures/frame_renders/logs/export150_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamesh_writeup/figures/frame_renders/logs/export150_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
FIG=$ROOT/experiments/dynamesh_writeup/figures/frame_renders
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd "$ROOT"
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="; nvidia-smi -L
$PY -u "$ROOT/render/export_frame_mesh.py" --frames 1-150 --no-glb \
  --run rung13_aligned_r4_s6_c742c0b3 \
  --alignment "$ROOT/experiments/lora_experiments/visibility/alignment/alignment.json" \
  --out-dir "$FIG/meshes"
echo "=== Done $(date) ==="
