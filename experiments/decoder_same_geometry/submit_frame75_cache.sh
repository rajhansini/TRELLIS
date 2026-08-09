#!/bin/bash
# Quick sanity: generate SLaT cache for frame 75 only (~5-7 min on L40S/A40).
# After this job completes, run test_frame75.py to verify LoRA render.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=slat_f75
#SBATCH --output=$LOG_DIR/slat_f75_%j.out
#SBATCH --error=$LOG_DIR/slat_f75_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|a40"
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --partition=general

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline
\$PY $SCRIPT_DIR/save_slat_cache.py --frame 75
echo "exit code: \$?"
EOF
