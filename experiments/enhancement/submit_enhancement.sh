#!/bin/bash
#SBATCH --job-name=enhancement_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results/slurm_%j.err

set -e

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

echo "=== GPU info ==="
nvidia-smi -L

echo "=== Starting enhancement sweep ==="
echo "  betas : 0 1 2 3 4 6 8 16"
echo "  frames: 1..150"
echo "  seed  : 6"

cd /net/projects/ranalab/rajhansini/TRELLIS

$PY /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step06_enhancement.py \
    --betas 0 1 2 3 4 6 8 16 \
    --start_frame 1 \
    --frames 150 \
    --base_seed 6

echo "=== Done ==="
