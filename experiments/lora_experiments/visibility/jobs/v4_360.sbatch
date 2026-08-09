#!/bin/bash
# v4 360 turntable — frozen TRELLIS | v4 | v4, no visibility mask.
#
# A REAL script file, not sbatch --wrap. --wrap executes under /bin/sh, which is
# dash on this cluster, and dash has no `set -o pipefail` — jobs 2143676/2143677
# died in 0 seconds with "Illegal option -o pipefail". A script with an explicit
# bash shebang is the fix, and it also makes the job reproducible and greppable.
#
# Submit:
#   sbatch experiments/lora_experiments/visibility/jobs/v4_360.sbatch
#SBATCH --job-name=v4_360
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/v4_360_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/v4_360_%j.log

set -eo pipefail

ROOT=/net/projects/ranalab/rajhansini/TRELLIS
VIS=${ROOT}/experiments/lora_experiments/visibility
LOGDIR=${VIS}/logs
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cd "${ROOT}"

echo "=================================================================="
echo "attempt at $(date)  host=$(hostname)  job=${SLURM_JOB_ID}"
echo "shell=${BASH_VERSION}"
nvidia-smi -L
echo "=================================================================="

echo ""
echo "### sweep=angle — frame 75, camera orbits 360 ###"
${PY} -u "${VIS}/render_vismask_orbit.py" \
  --run-id c85c888f --sweep angle --frame 75 \
  --n-angles 60 --elevation 15 --mask-mode none \
  2>&1 | tee -a "${LOGDIR}/v4_360_angle_${SLURM_JOB_ID}.log"

echo ""
echo "### sweep=both — frames 1..150 while the camera orbits ###"
${PY} -u "${VIS}/render_vismask_orbit.py" \
  --run-id c85c888f --sweep both \
  --elevation 15 --mask-mode none \
  2>&1 | tee -a "${LOGDIR}/v4_360_both_${SLURM_JOB_ID}.log"

echo ""
echo "=== outputs ==="
find "${VIS}/orbit_renders" -name '*.mp4' -printf '%p  %s bytes\n' 2>/dev/null || true
echo "=== Done at $(date) ==="
