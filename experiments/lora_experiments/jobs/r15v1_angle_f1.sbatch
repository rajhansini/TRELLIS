#!/bin/bash
# rung15-v1 videos: the temporal (dynamic texture) and the orbit (propagation test).
#SBATCH --job-name=r15v1_a1
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r15v1both_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/r15v1both_%j.log
set -eo pipefail
R15=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs/rung15v1_uniform_all_qkvo_r4_s6_2674ec70
E=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo ""; echo "######## BOTH (frame 1..150 AND camera orbits 360) ########"
${PY} -u "${E}/render_rung14v1_video.py" --run rung15v1_uniform_all_qkvo_r4_s6_2674ec70 --sweep angle --pin-frame 1 --n-angles 120 --no-gt --out-dir ${R15}/orbit_angle_f001 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"

echo "=== Done $(date) ==="
