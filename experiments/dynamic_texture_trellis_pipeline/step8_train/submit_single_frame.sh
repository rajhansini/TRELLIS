#!/bin/bash
#SBATCH --job-name=sf_lora
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results_single_frame/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results_single_frame/slurm_%j.err

# Usage:
#   sbatch submit_single_frame.sh                        # defaults: frame=75, epochs=30, lr=1e-4, seed=6
#   sbatch submit_single_frame.sh --frame 75 --epochs 50
#   sbatch --job-name=sf_f50 submit_single_frame.sh --frame 50 --epochs 30

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/pip

mkdir -p ../results_single_frame

echo "=== GPU info ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== Checking nvdiffrast compatibility ==="
set +e
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
print("nvdiffrast OK")
PYEOF
NVDIFFRAST_OK=$?
set -e

if [ "$NVDIFFRAST_OK" -ne 0 ]; then
    echo "=== nvdiffrast incompatible — rebuilding ==="
    SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
    echo "Detected SM: sm_${SM}"
    export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
    BUILD_DIR=$(mktemp -d)
    git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast" --depth 1 --quiet
    cd "$BUILD_DIR/nvdiffrast"
    $PIP install . --no-build-isolation -q
    cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
    rm -rf "$BUILD_DIR"
    echo "=== Re-checking nvdiffrast ==="
    SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
dr.RasterizeCudaContext()
print("nvdiffrast OK after rebuild")
PYEOF
fi

echo "=== Single-frame LoRA training (args: $@) ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u train_single_frame.py "$@"
EXIT=$?

if [ "$EXIT" -ne 0 ]; then
    echo "=== train_single_frame.py FAILED (exit $EXIT) ==="
    exit $EXIT
fi

echo "=== Done ==="
