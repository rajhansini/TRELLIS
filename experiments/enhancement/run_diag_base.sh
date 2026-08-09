#!/bin/bash
#SBATCH --job-name=diag_base_trellis
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_base/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_base/slurm_%j.log

mkdir -p /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_base

echo "=== diag_base job: $(date) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi -L)"

source /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/activate 2>/dev/null || \
    conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis

export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/projects/ranalab/rajhansini/TRELLIS

/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python \
    experiments/enhancement/diag_base_render.py \
    2>&1 | tee -a /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_base/run.log

echo "=== job done: $(date) ==="
