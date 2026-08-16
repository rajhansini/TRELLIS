#!/bin/bash
#SBATCH --job-name=base_spot_lava
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/base_spot_lava_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/base_spot_lava_%j.log
set -eo pipefail
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
  experiments/lora_experiments/rung22_kl_multi.py \
  --obj spot_lava \
  --frames-dir /net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video \
  --alignment experiments/lora_experiments/visibility/alignment_spot_lava/alignment.json \
  --ref-frame 75 \
  --rank 4 --epochs 30 --seed 6 --blocks all --targets kv \
  --tsample uniform --loss-region intersection \
  2>&1 | tee -a experiments/lora_experiments/logs/base_spot_lava_tee_${SLURM_JOB_ID}.log
echo "=== Done $(date) ==="
