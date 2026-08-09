#!/bin/bash
# Submit rung8 — intersection-loss colour LoRA (white-cast fix).
#
# Everything is logged, in three places:
#   1. slurm stdout+stderr  -> logs/rung8_<jobid>.log        (the master log)
#   2. explicit tee         -> logs/rung8_tee_<jobid>.log    (survives slurm quirks)
#   3. the script's own Tee -> runs/<label>/train.log        (survives requeue)
#   plus runs/<label>/logs/epoch_metrics.csv  (one row per epoch, machine-readable)
#        runs/<label>/logs/region_stats.json  (the |A|/|B|/IoU premise evidence)
#        runs/<label>/loss_history.json       (full history)
#
# Usage:
#   bash experiments/lora_experiments/visibility/submit_rung8.sh [--epochs 30]
#        [--rank 4] [--seed 6] [--lambda-reg 0.0]

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
BUILD_SCRIPT=${ROOT}/experiments/enhancement/build_renderers_A40.sh
LOGDIR=${VIS}/logs
mkdir -p "$LOGDIR"

EPOCHS=30
RANK=4
SEED=6
LAMBDA_REG=0.0
V4_RUN_ID=c85c888f

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --rank)       RANK="$2";       shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --lambda-reg) LAMBDA_REG="$2"; shift 2 ;;
        --v4-run-id)  V4_RUN_ID="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
SUBLOG=${LOGDIR}/rung8_submit_${STAMP}.log

{
echo "=== Rung 8 — intersection loss (white-cast fix) ==="
echo "  epochs     : ${EPOCHS}"
echo "  rank       : ${RANK}"
echo "  seed       : ${SEED}"
echo "  lambda_reg : ${LAMBDA_REG}  (0.0 = reproduces v4 effective behaviour)"
echo "  v4 run     : ${V4_RUN_ID}  (SLaT cache reused for comparability)"
echo "  logdir     : ${LOGDIR}"
echo "  submitted  : ${STAMP}"
echo ""

BUILD_JOB=$(sbatch \
    --job-name="build_a40" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGDIR}/rung8_build_%j.log" \
    --error="${LOGDIR}/rung8_build_%j.log" \
    --wrap="
set -e
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
bash ${BUILD_SCRIPT}
" | awk '{print $NF}')
echo "  build job  : ${BUILD_JOB}"

TRAIN_JOB=$(sbatch \
    --job-name="rung8_isect" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --requeue \
    --dependency=afterok:${BUILD_JOB} \
    --output="${LOGDIR}/rung8_%j.log" \
    --open-mode=append \
    --error="${LOGDIR}/rung8_%j.log" \
    --wrap="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
cd ${ROOT}
echo '=== GPU ==='; nvidia-smi -L; echo ''
${PY} -u ${VIS}/rung8_intersection_lora.py \
  --rank ${RANK} --epochs ${EPOCHS} --seed ${SEED} \
  --lambda-reg ${LAMBDA_REG} --v4-run-id ${V4_RUN_ID} \
  2>&1 | tee ${LOGDIR}/rung8_tee_\${SLURM_JOB_ID}.log
echo '=== Done ==='
" | awk '{print $NF}')
echo "  train job  : ${TRAIN_JOB}  (depends on build ${BUILD_JOB})"

echo ""
echo "LOGS"
echo "  master : ${LOGDIR}/rung8_${TRAIN_JOB}.log"
echo "  tee    : ${LOGDIR}/rung8_tee_${TRAIN_JOB}.log"
echo "  build  : ${LOGDIR}/rung8_build_${BUILD_JOB}.log"
echo "  submit : ${SUBLOG}"
echo "  train  : ${VIS}/runs/<label>/train.log"
echo "  csv    : ${VIS}/runs/<label>/logs/epoch_metrics.csv"
echo ""
echo "MONITOR"
echo "  squeue -u rajhansini"
echo "  tail -f ${LOGDIR}/rung8_${TRAIN_JOB}.log"
echo "  grep SHIFT ${LOGDIR}/rung8_${TRAIN_JOB}.log      # success criterion per epoch"
} 2>&1 | tee "$SUBLOG"
