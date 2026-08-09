#!/bin/bash
#SBATCH --job-name=test_orient
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY -u test_orientation.py
