#!/bin/bash
#SBATCH --job-name=verify_supervision
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python

SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u verify_supervision.py --frame 1 --frame 75 --frame 150 \
    2>&1 | tee ../results/verify_supervision.log

echo "=== Done. Results in results/supervision_check/ ==="
