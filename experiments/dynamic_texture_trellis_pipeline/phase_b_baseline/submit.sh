#!/bin/bash
#SBATCH --job-name=phase_b
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/phase_b_baseline

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python

echo "=== Phase B: flickering baseline (lambda_curr=1.0, lambda_next=0.0) ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u run.py --lambda_curr 1.0 --lambda_next 0.0 --tag flickering \
    2>&1 | tee ../../results/phase_b_flickering.log

echo ""
echo "=== Phase C1: lambda_curr=0.9, lambda_next=0.1 ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u run.py --lambda_curr 0.9 --lambda_next 0.1 --tag blend_01 \
    2>&1 | tee ../../results/phase_b_blend_01.log

echo ""
echo "=== Phase C2: lambda_curr=0.5, lambda_next=0.5 ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u run.py --lambda_curr 0.5 --lambda_next 0.5 --tag blend_05 \
    2>&1 | tee ../../results/phase_b_blend_05.log

echo "=== All done ==="
