#!/bin/bash
# Option B — visibility masking. Submits the three stages as dependent slurm jobs.
#
#   stage 1  probe    compute_visibility_mask.py   (~10 min, 150 fwd+bwd)
#   stage 2  orbit    render_vismask_orbit.py      (~5 min, NO retraining)
#   stage 3  train    rung7_vismask_lora.py        (~3 h, OPTIONAL ablation)
#
# Usage:
#   bash experiments/lora_experiments/visibility/submit_visibility.sh [options]
#
#     --stages probe,orbit        which stages to run (default: probe,orbit)
#     --stages probe,orbit,train  include the optional retrain
#     --mask-mode soft|hard|none  default soft
#     --delta-channels all|rgb    all = v4 behaviour (albedo+normals); rgb = albedo only
#     --run-id c85c888f           v4 run providing slat_cache + lora_best
#     --frame 75                  SLaT frame for the single-frame turntable
#     --n-angles 60               turntable steps
#     --epochs 30                 retrain epochs
#
# Every job tees into experiments/lora_experiments/visibility/logs/.

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
BUILD_SCRIPT=${ROOT}/experiments/enhancement/build_renderers_A40.sh
LOGDIR=${VIS}/logs
mkdir -p "$LOGDIR"

STAGES=probe,orbit
MASK_MODE=soft
DELTA_CHANNELS=all
RUN_ID=c85c888f
FRAME=75
N_ANGLES=60
ELEVATION=15
EPOCHS=30
RANK=4
SEED=6
LAMBDA_REG=0.01

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)     STAGES="$2";     shift 2 ;;
        --mask-mode)  MASK_MODE="$2";  shift 2 ;;
        --delta-channels) DELTA_CHANNELS="$2"; shift 2 ;;
        --run-id)     RUN_ID="$2";     shift 2 ;;
        --frame)      FRAME="$2";      shift 2 ;;
        --n-angles)   N_ANGLES="$2";   shift 2 ;;
        --elevation)  ELEVATION="$2";  shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --rank)       RANK="$2";       shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --lambda-reg) LAMBDA_REG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

has_stage () { [[ ",${STAGES}," == *",$1,"* ]]; }

echo "=== Option B — visibility masking ==="
echo "  stages    : ${STAGES}"
echo "  mask mode : ${MASK_MODE}"
echo "  channels  : ${DELTA_CHANNELS}"
echo "  v4 run    : ${RUN_ID}"
echo "  logs      : ${LOGDIR}"
echo ""

ENVSETUP="
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
cd ${ROOT}
"

# ── stage 0: rebuild renderers for whatever GPU arch we land on ───────────────
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
echo "  build job : ${BUILD_JOB}"

DEP=${BUILD_JOB}

# ── stage 1: visibility probe ─────────────────────────────────────────────────
if has_stage probe; then
PROBE_JOB=$(sbatch \
    --job-name="vis_probe" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:00:00 \
    --requeue \
    --dependency=afterok:${DEP} \
    --output="${LOGDIR}/vis_probe_%j.log" \
    --error="${LOGDIR}/vis_probe_%j.log" \
    --wrap="
set -e
${ENVSETUP}
nvidia-smi -L
${PY} -u ${VIS}/compute_visibility_mask.py --run-id ${RUN_ID} \
  2>&1 | tee ${LOGDIR}/vis_probe_\${SLURM_JOB_ID}_tee.log
" | awk '{print $NF}')
echo "  probe job : ${PROBE_JOB}"
DEP=${PROBE_JOB}
fi

# ── stage 2: masked turntable (no retraining) ─────────────────────────────────
if has_stage orbit; then
ORBIT_JOB=$(sbatch \
    --job-name="vis_orbit" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:30:00 \
    --requeue \
    --dependency=afterok:${DEP} \
    --output="${LOGDIR}/vis_orbit_%j.log" \
    --error="${LOGDIR}/vis_orbit_%j.log" \
    --wrap="
set -e
${ENVSETUP}
${PY} -u ${VIS}/render_vismask_orbit.py \
  --run-id ${RUN_ID} --sweep angle --frame ${FRAME} \
  --n-angles ${N_ANGLES} --elevation ${ELEVATION} --mask-mode ${MASK_MODE} \
  --delta-channels ${DELTA_CHANNELS} \
  2>&1 | tee ${LOGDIR}/vis_orbit_angle_\${SLURM_JOB_ID}_tee.log
${PY} -u ${VIS}/render_vismask_orbit.py \
  --run-id ${RUN_ID} --sweep both \
  --elevation ${ELEVATION} --mask-mode ${MASK_MODE} \
  --delta-channels ${DELTA_CHANNELS} \
  2>&1 | tee ${LOGDIR}/vis_orbit_both_\${SLURM_JOB_ID}_tee.log
" | awk '{print $NF}')
echo "  orbit job : ${ORBIT_JOB}"
DEP=${ORBIT_JOB}
fi

# ── stage 3: optional retrain ─────────────────────────────────────────────────
if has_stage train; then
TRAIN_JOB=$(sbatch \
    --job-name="r7_vismask" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=05:00:00 \
    --requeue \
    --dependency=afterok:${DEP} \
    --output="${LOGDIR}/rung7_vismask_%j.log" \
    --error="${LOGDIR}/rung7_vismask_%j.log" \
    --wrap="
set -e
${ENVSETUP}
nvidia-smi -L
${PY} -u ${VIS}/rung7_vismask_lora.py \
  --rank ${RANK} --epochs ${EPOCHS} --seed ${SEED} \
  --lambda-reg ${LAMBDA_REG} --mask-mode ${MASK_MODE} --v4-run-id ${RUN_ID} \
  --delta-channels ${DELTA_CHANNELS} \
  2>&1 | tee ${LOGDIR}/rung7_vismask_\${SLURM_JOB_ID}_tee.log
" | awk '{print $NF}')
echo "  train job : ${TRAIN_JOB}"
fi

echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo "  tail -f ${LOGDIR}/vis_probe_*.log"
echo "  tail -f ${LOGDIR}/vis_orbit_*.log"
