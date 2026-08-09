#!/bin/bash
# Submit canonical-geometry render jobs for all rung5 runs.
# Each job: frozen flow → SLaTs → LoRA decoder with canonical SDF+deform → render → video.

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/decoder_same_geometry/render_canonical_geom.py
RUNS_DIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs
OUT_BASE=/net/projects/ranalab/rajhansini/TRELLIS/experiments/decoder_same_geometry/rendered_videos
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/decoder_same_geometry/logs
mkdir -p "$OUT_BASE" "$LOGDIR"

RUNS=(
  rung5_50_r4_s6_da985400
  rung5_51_r4_s6_09d4d58b
  rung5_52_r4_s6_3d7617b4
  rung5_53_early_r4_s6_05dbf396
  rung5_53_mid_r4_s6_40243e3f
  rung5_53_late_r4_s6_1f298d4d
  rung5_54_r4_s6_31e5b3e8
  rung5_55_r4_s6_e8470ed6
  rung5_55_r8_s6_a65168bf
  rung5_55_r16_s6_9c47bd2e
)

for RUN_NAME in "${RUNS[@]}"; do
  RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
  OUT_DIR="${OUT_BASE}/${RUN_NAME}"
  mkdir -p "$OUT_DIR"

  JOB_ID=$(sbatch \
    --job-name="cgeom_${RUN_NAME:0:18}" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:30:00 \
    --output="${LOGDIR}/${RUN_NAME}_%j.log" \
    --error="${LOGDIR}/${RUN_NAME}_%j.log" \
    --wrap="
set -e
export SPCONV_ALGO=native ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} -u ${SCRIPT} --run-dir ${RUN_DIR} --out-dir ${OUT_DIR} --fps 12
" | awk '{print $NF}')
  echo "${RUN_NAME}: job=${JOB_ID}"
done

echo ""
echo "Videos → ${OUT_BASE}/<run_name>/<run_name>_canonical_geom.mp4"
