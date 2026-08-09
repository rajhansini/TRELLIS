#!/bin/bash
# Submit save_slat_cache.py on an L40S or A40 node.
# Must run on L40S/A40 — RTX 2080 Ti gives wrong N_vox.
# Output: experiments/decoder_same_geometry/rung5_slat_cache.npz
# (shared by all 10 rung5 runs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=save_slat_cache
#SBATCH --output=$LOG_DIR/save_slat_cache_%j.out
#SBATCH --error=$LOG_DIR/save_slat_cache_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|a40"
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=general

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline
\$PY $SCRIPT_DIR/save_slat_cache.py
echo "exit code: \$?"
EOF
