#!/bin/bash
#SBATCH --job-name=final_f75
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/render/out/final_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/render/out/final_%j.log
set -eo pipefail
R=/net/projects/ranalab/rajhansini/TRELLIS/render
O=$R/out/final; mkdir -p "$O"
BLENDER=${HOME}/blender-3.2.1-linux-x64/blender
cd /net/projects/ranalab/rajhansini/TRELLIS
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L

# orientation solved empirically: our mesh is +Z up (same as Blender), so NO x-flip.
# rz=135 gives the 3/4 view with lid up, spout right, handle left.
render () {  # $1 mesh  $2 out  $3 cam_x $4 cam_y $5 cam_z  $6 scale
  "$BLENDER" "$R/blend_files/blank.blend" --background \
    --python "$R/render_single_mesh.py" -- \
    "$1" "$R/config/dynamic_config.json" "$2" \
    --background-color-hex "#FFFFFFFF" \
    --resolution 1000 1000 --samples 128 \
    --object-rotation 0 0 135 --object-location 0 0 0 --object-scale "$6" \
    --camera-location "$3" "$4" "$5" --camera-rotation 75 -5 130 \
    >/dev/null 2>&1
  echo "  wrote $(basename $2)"
}

for ARM in frozen result; do
  render "$R/out/f0075/${ARM}_f0075.ply" "$O/${ARM}_hi.png"  3.2 2.4 3.0 2.4
  render "$R/out/f0075/${ARM}_f0075.ply" "$O/${ARM}_eye.png" 3.6 2.7 1.9 2.4
done
ls -la "$O"
echo "=== Done $(date) ==="
