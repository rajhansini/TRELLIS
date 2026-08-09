#!/bin/bash
#SBATCH --job-name=build_matrices
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/artifacts/slurm_build_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/artifacts/slurm_build_%j.err

set -e

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

echo "=== GPU info ==="
nvidia-smi -L

echo "=== Building 20 alignment matrices (phase C v1/v2, phase D v1/v2) ==="

cd /net/projects/ranalab/rajhansini/TRELLIS

$PY /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/step05b_build_matrices.py \
    --phase all --variant all

echo "=== Done ==="
