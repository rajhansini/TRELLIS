#!/bin/bash
# One-time: cache u2net foreground masks for the 150 GT frames.
# CPU only -- u2net runs on CPU here, ~21 s/frame, ~53 min total.
# Both E1 (rung16) and E2 (rung17) load this cache, so it is computed once.
#SBATCH --job-name=rembg_msk
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/rembg_msk_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/logs/rembg_msk_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
LEX=${ROOT}/experiments/lora_experiments
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
mkdir -p "${LEX}/logs"
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
${PY} -u "${LEX}/precompute_rembg_masks.py" \
  2>&1 | tee -a "${LEX}/logs/rembg_msk_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
