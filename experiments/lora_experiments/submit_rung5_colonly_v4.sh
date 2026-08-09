#!/bin/bash
# Submit rung5 color-only v4 (out_layer hook LoRA) decoder training.
# Usage: bash submit_rung5_colonly_v4.sh [--rank R] [--epochs E] [--seed S] [--lambda-reg L]

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung5_colonly_lora_v4.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

RANK=4
EPOCHS=30
SEED=6
LAMBDA_REG=0.01

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)       RANK="$2";       shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --lambda-reg) LAMBDA_REG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Rung 5 Color-Only Decoder LoRA v4 (out_layer hook) ==="
echo "  rank=$RANK  epochs=$EPOCHS  seed=$SEED  lambda_reg=$LAMBDA_REG"
echo ""

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
echo "  build job : $BUILD_JOB"

TRAIN_JOB=$(sbatch \
    --job-name="r5_outlayer" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --requeue \
    --dependency=afterok:${BUILD_JOB} \
    --output="${LOGDIR}/rung5_outlayer_%j.log" \
    --error="${LOGDIR}/rung5_outlayer_%j.log" \
    --wrap="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
echo '=== GPU info ==='
nvidia-smi -L
echo ''
cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} -u ${SCRIPT} --rank ${RANK} --epochs ${EPOCHS} --seed ${SEED} --lambda-reg ${LAMBDA_REG}
echo '=== Done ==='
" | awk '{print $NF}')
echo "  train job : $TRAIN_JOB  (depends on build $BUILD_JOB)"

echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo "  tail -f ${LOGDIR}/rung5_outlayer_${TRAIN_JOB}.log"
