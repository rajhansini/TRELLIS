#!/bin/bash
#SBATCH --job-name=render150
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamesh_writeup/figures/frame_renders/logs/render150_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamesh_writeup/figures/frame_renders/logs/render150_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
R=$ROOT/render
FIG=$ROOT/experiments/dynamesh_writeup/figures/frame_renders
GT=/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150
BLENDER=$HOME/blender-3.2.1-linux-x64/blender
cd "$ROOT"
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="; nvidia-smi -L

# settled recipe: eye-level camera, +Z up so no x-flip, opaque white background
render () {
  "$BLENDER" "$R/blend_files/blank.blend" --background \
    --python "$R/render_single_mesh.py" -- \
    "$1" "$R/config/dynamic_config.json" "$2" \
    --background-color-hex "#FFFFFFFF" \
    --resolution 1000 1000 --samples 128 \
    --object-rotation 0 0 135 --object-location 0 0 0 --object-scale 2.4 \
    --camera-location 3.6 2.7 1.9 --camera-rotation 75 -5 130 \
    >/dev/null 2>&1
}

for i in $(seq -w 1 150); do
  F=$(echo $i | sed 's/^0*//'); F=$(printf "%04d" ${F:-0})
  [ -f "$FIG/baseline/frame_$F.png" ] || render "$FIG/meshes/frozen_f$F.ply" "$FIG/baseline/frame_$F.png"
  [ -f "$FIG/ours/frame_$F.png" ]     || render "$FIG/meshes/result_f$F.ply" "$FIG/ours/frame_$F.png"
  cp -n "$GT/frame_$F.png" "$FIG/gt/frame_$F.png" 2>/dev/null || true
  [ $((10#$F % 10)) -eq 0 ] && echo "  $F/150  $(date +%T)"
done

echo "--- counts"; for d in gt baseline ours; do echo "  $d: $(ls $FIG/$d/*.png 2>/dev/null | wc -l)"; done
echo "=== Done $(date) ==="
