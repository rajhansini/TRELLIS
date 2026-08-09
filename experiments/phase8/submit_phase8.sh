#!/bin/bash
#SBATCH --job-name=phase8_slat
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/results/phase8/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/results/phase8/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

mkdir -p experiments/results/phase8

SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY experiments/phase8/phase8.py --k 1 --alpha 1.0 --mode all \
    2>&1 | tee experiments/results/phase8/run_k1_a1.log

echo "=== Phase 8 done ==="
