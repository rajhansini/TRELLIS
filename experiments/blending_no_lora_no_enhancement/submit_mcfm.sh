#!/bin/bash
#SBATCH --job-name=mcfm_blend
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/blending_no_lora_no_enhancement/slurm_logs/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/blending_no_lora_no_enhancement/slurm_logs/slurm_%j.err

# Usage:
#   sbatch submit_mcfm.sh --mode v2_D --seed 6 --start_frame 1 --frames 150
#   sbatch submit_mcfm.sh --mode v3_C --seed 6 --start_frame 1 --frames 150
#   sbatch submit_mcfm.sh --mode v3_D --seed 6 --start_frame 1 --frames 150

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/blending_no_lora_no_enhancement

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip

echo "=== GPU info ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== Checking nvdiffrast compatibility ==="
set +e
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
dr.RasterizeCudaContext()
print("nvdiffrast OK")
PYEOF
NVDIFFRAST_OK=$?
set -e

if [ "$NVDIFFRAST_OK" -ne 0 ]; then
    echo "=== nvdiffrast incompatible — rebuilding for this GPU ==="
    SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
    echo "Detected SM: sm_${SM}"
    export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
    BUILD_DIR=$(mktemp -d)
    git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast" --depth 1 --quiet
    cd "$BUILD_DIR/nvdiffrast"
    $PIP install . --no-build-isolation -q
    cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/blending_no_lora_no_enhancement
    rm -rf "$BUILD_DIR"
    echo "=== Re-checking nvdiffrast ==="
    SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
dr.RasterizeCudaContext()
print("nvdiffrast OK after rebuild")
PYEOF
fi

echo "=== Running MCFM blending (args: $@) ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u run.py "$@"
EXIT=$?

if [ "$EXIT" -ne 0 ]; then
    echo "=== run.py FAILED (exit $EXIT) ==="
    exit $EXIT
fi

echo "=== Done ==="
