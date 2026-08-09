#!/bin/bash
# Submit render jobs for all canonical runs — produces GT|frozen|LoRA comparison videos.
# Each job: load pipeline → DINOv2 encode 150 frames → denoise SLaTs → render → ffmpeg video.
# Estimated runtime: ~30-45 min per run (GPU-bound).

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/render_run_video.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
RUNS_DIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs
OUT_BASE=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rendered_videos
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$OUT_BASE" "$LOGDIR"

# ── shared nvdiffrast build ───────────────────────────────────────────────────
echo "=== Build job ==="
BUILD_JOB=$(sbatch \
    --job-name="build_a40" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGDIR}/build_a40_%j.log" \
    --error="${LOGDIR}/build_a40_%j.log" \
    --wrap="
set -e
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
bash ${BUILD_SCRIPT}
" | awk '{print $NF}')
echo "  build: ${BUILD_JOB}"
echo ""

# ── canonical runs to render ──────────────────────────────────────────────────
declare -a RUNS=(
    "rung2_late_r4_s6_4a3dea1d"
    "rung3_late_r4_s6_0534fdf2"
    "rung5_53_mid_r4_s6_40243e3f"
    "rung5_54_r4_s6_31e5b3e8"
    "rung1_single_path_r4_s6"
)

echo "=== Render jobs ==="
for RUN_NAME in "${RUNS[@]}"; do
    RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
    OUT_DIR="${OUT_BASE}/${RUN_NAME}"
    LOG="${LOGDIR}/render_${RUN_NAME}_%j.log"

    JOB_ID=$(sbatch \
        --job-name="render_${RUN_NAME:0:20}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=01:30:00 \
        --dependency=afterok:${BUILD_JOB} \
        --output="${LOG}" \
        --error="${LOG}" \
        --wrap="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
echo '=== GPU ==='
nvidia-smi -L
echo ''
echo '=== Rendering: ${RUN_NAME} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} -u ${SCRIPT} \
    --run-dir ${RUN_DIR} \
    --out-dir ${OUT_DIR} \
    --fps 12
echo '=== Done ==='
" | awk '{print $NF}')
    echo "  ${RUN_NAME}: job=${JOB_ID}"
done

echo ""
echo "=== All render jobs submitted ==="
echo "Monitor:"
echo "  squeue -u rajhansini | grep render"
echo "Videos will be at: ${OUT_BASE}/<run_name>/<run_name>_comparison.mp4"
