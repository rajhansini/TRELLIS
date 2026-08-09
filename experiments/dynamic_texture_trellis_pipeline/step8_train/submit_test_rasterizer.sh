#!/bin/bash
#SBATCH --job-name=test_rasterizer
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/test_rasterizer_%j.log

cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY -u test_rasterizer.py
