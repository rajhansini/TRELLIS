#!/bin/bash
#SBATCH --job-name=phase_b_f77
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/phase_b_baseline

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python

echo "=== Phase B: single frame 77, flickering baseline ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u run.py \
        --lambda_curr 1.0 --lambda_next 0.0 \
        --start_frame 77 --frames 1 \
        --tag flickering \
    2>&1 | tee ../../results/phase_b_f77.log

echo "=== Done ==="
