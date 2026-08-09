#!/bin/bash
# Rung 11 — rung5_colonly + visibility masking. Two steps, one job.
#
#   step 1  compute_visibility_mask.py  (EXISTING FILE, RUN UNMODIFIED)
#           per-fine-voxel gradient scores -> visibility/masks/visibility.npz
#           skipped if the npz already exists, so a requeue does not redo it
#   step 2  rung11_vismask_colonly_lora.py  --mask-mode soft
#
# Real script file (sbatch --wrap runs /bin/sh = dash, which has no pipefail).
# --requeue + --open-mode=append: rung11 resumes from its latest epoch ckpt.
#SBATCH --job-name=rung11_vism
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung11_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/visibility/logs/rung11_%j.log

set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
VIS=${LEX}/visibility
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${VIS}/logs" "${VIS}/masks"
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi -L

MASK_NPZ="${VIS}/masks/visibility.npz"
if [ -f "${MASK_NPZ}" ]; then
  echo "[STEP 1] mask already exists at ${MASK_NPZ} — skipping"
else
  echo "[STEP 1] computing visibility mask (unmodified compute_visibility_mask.py)"
  ${PY} -u "${VIS}/compute_visibility_mask.py" \
    --run-dir "${LEX}/runs/rung5_colonly_r4_s6_5d6b1700" \
    2>&1 | tee -a "${VIS}/logs/rung11_mask_${SLURM_JOB_ID}.log"
fi

echo "[STEP 2] training rung11"
${PY} -u "${LEX}/rung11_vismask_colonly_lora.py" \
  --rank 4 --epochs 30 --seed 6 --mask-mode soft \
  2>&1 | tee -a "${VIS}/logs/rung11_tee_${SLURM_JOB_ID}.log"

echo "=== Done $(date) ==="
