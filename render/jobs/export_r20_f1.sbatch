#!/bin/bash
#SBATCH --job-name=exp20_f1
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/render/logs/exp20_f1_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/render/logs/exp20_f1_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export SPCONV_ALGO=native ATTN_BACKEND=xformers CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd "${ROOT}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
${PY} -u render/export_xattn_mesh.py \
  --run rung20_intersection_uniform_all_kv_r4_s6_c7a82c89 --frame 1 \
  2>&1 | tee -a "${ROOT}/render/logs/exp20_f1_tee_${SLURM_JOB_ID}.log"
echo "=== Done $(date) ==="
