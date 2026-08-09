#!/bin/bash
set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step07d_render_best.py
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/slurm_logs
mkdir -p "$LOGDIR"

echo "=== Submitting render jobs ==="

for ALPHA in 0.1 0.25 0.5; do
    ALPHA_STR=$(echo $ALPHA | sed 's/\./p/')

    WRAP="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

echo '=== Render alpha=${ALPHA} ==='
nvidia-smi -L

cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} ${SCRIPT} --alpha ${ALPHA} --mode v2_C

echo '[RENDER] done.'
"

    JOB=$(sbatch \
        --job-name="rnd_a${ALPHA_STR}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --requeue \
        --output="${LOGDIR}/render_alpha${ALPHA_STR}_%j.log" \
        --error="${LOGDIR}/render_alpha${ALPHA_STR}_%j.log" \
        --wrap="$WRAP" | awk '{print $NF}')

    echo "  alpha=${ALPHA}  job=${JOB}  log=${LOGDIR}/render_alpha${ALPHA_STR}_${JOB}.log"
done

echo "=== All 3 render jobs submitted ==="
echo "Monitor: squeue -u rajhansini"
