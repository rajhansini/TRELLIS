#!/bin/bash
# Smoke test: glb -> ply -> Blender/Cycles render with shadow.
#SBATCH --job-name=blender_test
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/render/out/test_render_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/render/out/test_render_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
R=${ROOT}/render
BLENDER=${HOME}/blender-3.2.1-linux-x64/blender
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo "--- 1. glb -> ply (textured source, forces the sampling path)"
${PY} "${R}/glb_to_ply.py" \
  /net/projects/ranalab/rajhansini/multi_iSeg/meshes/meshestotrain/armadillo.glb \
  "${R}/out/armadillo.ply" --normalize

echo "--- 2. Blender render with shadow"
"${BLENDER}" "${R}/blend_files/blank.blend" --background \
  --python "${R}/render_single_mesh.py" -- \
  "${R}/out/armadillo.ply" "${R}/config/dynamic_config.json" "${R}/out/armadillo_render.png" \
  --resolution 800 800 \
  --object-rotation 90 0 36 --object-location 0 0 0 --object-scale 1.8 \
  --camera-location 3.2 2.4 3.0 --camera-rotation 75 -5 130 \
  --samples 32

echo "--- result"
ls -la "${R}/out/"
echo "=== Done $(date) ==="
