#!/bin/bash
# Pinned-alpha LoRA sweep — step07d, 3 alpha values, v2_C mode, 15 epochs each.
# Each alpha run is one independent job (no chaining needed at 15 epochs, ~3.25h each).
#
# Fixes vs step07c:
#   - kaiming_uniform A init (||A||~1.6 not 16)
#   - (lora_alpha/rank)=1.0 scaling on LoRA output
#   - alpha fixed float, not nn.Parameter, no ALPHA_REG
#   - weight_decay=1e-2 on A and B
#
# Job chain:
#   BUILD_JOB : rebuild nvdiffrast for sm_89 (L40S) on one L40S node
#   TRAIN_JOBS: 3 training jobs, each --dependency=afterok:BUILD_JOB
#
# Logs:
#   Build Slurm : slurm_logs/build_L40S_{jobid}.log
#   Python Tee  : results_mcfm_v2_C_d_alpha{0p10,0p25,0p50}_seed6/train_v2_C_alpha*.log
#   Train Slurm : slurm_logs/lora_pinned_alpha{val}_{jobid}.log

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07d_lora_pinned_alpha.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_L40S.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

MODE=v2_C
EPOCHS=15
RANK=4
LR=1e-4
LOSS_SCALE=4096

echo "=== Step 1: Submit nvdiffrast rebuild job for L40S (sm_89) ==="

BUILD_JOB=$(sbatch \
    --job-name="build_L40S" \
    --partition=general \
    --gres=gpu:L40S:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGDIR}/build_L40S_%j.log" \
    --error="${LOGDIR}/build_L40S_%j.log" \
    --wrap="
set -e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
bash ${BUILD_SCRIPT}
" | awk '{print $NF}')

echo "  Build job : ${BUILD_JOB}"
echo "  Build log : ${LOGDIR}/build_L40S_${BUILD_JOB}.log"
echo ""

echo "=== Step 2: Submit 3 training jobs with --dependency=afterok:${BUILD_JOB} ==="
echo "  mode=${MODE}, epochs=${EPOCHS}, alphas=[0.1, 0.25, 0.5]"
echo ""

for ALPHA in 0.1 0.25 0.5; do

    ALPHA_STR=$(echo $ALPHA | sed 's/\./p/')

    WRAP="
set -e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

echo '=== GPU info ==='
nvidia-smi -L
echo ''
echo '=== step07d: mode=${MODE} alpha=${ALPHA} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} \
    --mode ${MODE} \
    --alpha ${ALPHA} \
    --epochs ${EPOCHS} \
    --rank ${RANK} \
    --lr ${LR} \
    --loss_scale ${LOSS_SCALE}

echo '=== DONE: mode=${MODE} alpha=${ALPHA} ==='
"

    JOB=$(sbatch \
        --job-name="lora_a${ALPHA_STR}" \
        --partition=general \
        --gres=gpu:L40S:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        --dependency=afterok:${BUILD_JOB} \
        --output="${LOGDIR}/lora_pinned_alpha${ALPHA_STR}_%j.log" \
        --error="${LOGDIR}/lora_pinned_alpha${ALPHA_STR}_%j.log" \
        --wrap="$WRAP" | awk '{print $NF}')

    echo "  alpha=${ALPHA}  job=${JOB}  (waits on build job ${BUILD_JOB})"
    echo "  Slurm log : ${LOGDIR}/lora_pinned_alpha${ALPHA_STR}_${JOB}.log"
    echo "  Python log: /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_d_alpha${ALPHA_STR}_seed6/train_${MODE}_alpha${ALPHA_STR}_epochs${EPOCHS}.log"
    echo ""

done

echo "=== All 4 jobs submitted (1 build + 3 train) ==="
echo ""
echo "Monitor with:"
echo "  squeue -u rajhansini"
echo ""
echo "Read build log (live):"
echo "  tail -f ${LOGDIR}/build_L40S_${BUILD_JOB}.log"
echo ""
echo "Read Python logs (live, after build completes):"
for ALPHA_STR in 0p10 0p25 0p50; do
    echo "  tail -f /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_${MODE}_d_alpha${ALPHA_STR}_seed6/train_${MODE}_alpha${ALPHA_STR}_epochs${EPOCHS}.log"
done
