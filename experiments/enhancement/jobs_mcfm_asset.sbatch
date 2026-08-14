#!/bin/bash
#SBATCH --job-name=t1mcfm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/logs_mcfm/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/logs_mcfm/%x_%j.log
set -eo pipefail
R=/net/projects/ranalab/rajhansini/TRELLIS
E=${R}/experiments/enhancement
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SPCONV_ALGO=native ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
cd ${R}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

# PHASE=c|d  VARIANT=v2|v3  ASSET=<name>  NF=<frames>
${PY} -u ${E}/step06b_asset.py \
  --betas 0 \
  --start_frame ${SF:-1} --frames ${NF} --base_seed 6 \
  --phase ${PHASE} --variant ${VARIANT} \
  --asset ${ASSET} \
  --gt-dir /net/projects/ranalab/rajhansini/TRELLIS.2/data/${ASSET}/frames_from_video \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]$"

echo "=== Done $(date) ==="
