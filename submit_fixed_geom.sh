#!/bin/bash
#SBATCH --job-name=trellis_fixed_geom
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/mvadaptornew/mvadaptorresults/trellis_fixed_geom/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/mvadaptornew/mvadaptorresults/trellis_fixed_geom/slurm_%j.err

set -e

OUTPUT_DIR=/net/projects/ranalab/rajhansini/mvadaptornew/mvadaptorresults/trellis_fixed_geom
FRAMES_DIR=/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150

TRELLIS_PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
DYN_PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

mkdir -p $OUTPUT_DIR

echo "=== Step 1: TRELLIS fixed-geometry texturing ==="
$TRELLIS_PY /net/projects/ranalab/rajhansini/TRELLIS/run_trellis_fixed_geom.py \
    --frames_dir  $FRAMES_DIR \
    --output_dir  $OUTPUT_DIR \
    --canonical_frame 75 \
    --seed 42

echo "=== Step 2: Convert GLBs to OBJ ==="
$DYN_PY /net/projects/ranalab/rajhansini/dynamic_texture/src/convert_glbs_to_obj.py \
    --frames_dir $OUTPUT_DIR

echo "=== Step 3: Render from canonical camera angles ==="
$DYN_PY /net/projects/ranalab/rajhansini/dynamic_texture/src/render_mvadaptor_render_textured_glb.py \
    --frames_dir $OUTPUT_DIR \
    --global_normalize

echo "=== Step 4: Stitch front-view video ==="
ffmpeg -y -framerate 30 \
    -pattern_type glob -i "$OUTPUT_DIR/frame_*/renders/front.png" \
    -vf scale=512:512 -c:v libx264 -pix_fmt yuv420p \
    $OUTPUT_DIR/trellis_fixed_geom_front.mp4

echo "=== Done. Video: $OUTPUT_DIR/trellis_fixed_geom_front.mp4 ==="
