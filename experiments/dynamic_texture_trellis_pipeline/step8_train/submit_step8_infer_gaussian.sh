#!/bin/bash
#SBATCH --job-name=step8_infer_gs
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python

mkdir -p ../results/inference_frames_gaussian

echo "=== GPU info ==="
nvidia-smi -L || true

echo "=== Gaussian Inference ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u infer_gaussian.py 2>&1 | tee ../results/infer_gaussian.log
INFER_EXIT=${PIPESTATUS[0]}
if [ "$INFER_EXIT" -ne 0 ]; then
    echo "=== infer_gaussian.py FAILED (exit $INFER_EXIT) ==="
    exit $INFER_EXIT
fi

echo "=== Done ==="
