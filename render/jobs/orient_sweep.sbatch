#!/bin/bash
# Find the right object-rotation for our teapot, on a WHITE background.
#SBATCH --job-name=orient
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/render/out/orient_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/render/out/orient_%j.log
set -eo pipefail
R=/net/projects/ranalab/rajhansini/TRELLIS/render
O=$R/out/orient; mkdir -p "$O"
BLENDER=${HOME}/blender-3.2.1-linux-x64/blender
MESH=$R/out/f0075/result_f0075.ply
cd /net/projects/ranalab/rajhansini/TRELLIS
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L

# our world: up = -z, object faces -y.  Blender: +Z up.
# so a 180 deg flip about X maps our up onto Blender's; sweep azimuth to find the face.
for RX in 0; do
for RZ in 0 45 90 135 180 225 270 315; do
  TAG="rx${RX}_rz${RZ}"
  "$BLENDER" "$R/blend_files/blank.blend" --background \
    --python "$R/render_single_mesh.py" -- \
    "$MESH" "$R/config/dynamic_config.json" "$O/$TAG.png" \
    --background-color-hex "#FFFFFFFF" \
    --resolution 400 400 --samples 16 \
    --object-rotation $RX 0 $RZ --object-location 0 0 0 --object-scale 1.8 \
    --camera-location 3.2 2.4 3.0 --camera-rotation 75 -5 130 \
    >/dev/null 2>&1
  echo "  rendered $TAG"
done
done
ls -la "$O"
echo "=== Done $(date) ==="
