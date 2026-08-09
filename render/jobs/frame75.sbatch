#!/bin/bash
# frame 75: decode rung13 -> glb + ply -> Blender render with shadow, both arms.
#SBATCH --job-name=f75_render
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/render/out/f75_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/render/out/f75_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
R=${ROOT}/render
O=${R}/out/f0075
BLENDER=${HOME}/blender-3.2.1-linux-x64/blender
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo "### 1. decode frame 75 -> glb + ply"
${PY} -u "${R}/export_frame_mesh.py" --frame 75 \
  --run rung13_aligned_r4_s6_c742c0b3 \
  --alignment "${ROOT}/experiments/lora_experiments/visibility/alignment/alignment.json" \
  --out-dir "${O}"

echo "### 2. glb -> ply through the converter (exercises the new path)"
${PY} "${R}/glb_to_ply.py" "${O}/result_f0075.glb" "${O}/result_f0075_viaglb.ply"

echo "### 3. Blender render, both arms, identical camera and lighting"
for TAG in frozen_f0075 result_f0075; do
  echo "--- rendering ${TAG}"
  "${BLENDER}" "${R}/blend_files/blank.blend" --background \
    --python "${R}/render_single_mesh.py" -- \
    "${O}/${TAG}.ply" "${R}/config/dynamic_config.json" "${O}/${TAG}_blender.png" \
    --resolution 1000 1000 \
    --object-rotation 90 0 36 --object-location 0 0 0 --object-scale 1.8 \
    --camera-location 3.2 2.4 3.0 --camera-rotation 75 -5 130 \
    --samples 64 2>&1 | grep -vE "^Fra:|pulse|gvfs" | tail -4
done

echo "### result"; ls -la "${O}"
echo "=== Done $(date) ==="
