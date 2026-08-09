#!/bin/bash
#SBATCH --job-name=rebuild_renderers
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/results/phase8/build_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/results/phase8/build_%j.err

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip

echo "=== GPU info ==="
nvidia-smi -L

# Detect SM version
SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
echo "Detected SM version: sm_${SM}"
export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

BUILD_DIR=/tmp/renderer_build_$$
mkdir -p $BUILD_DIR

# ── 1. Rebuild diff-gaussian-rasterization (mip-splatting) ────────────────────
echo ""
echo "=== Rebuilding diff-gaussian-rasterization for sm_${SM} ==="
git clone https://github.com/autonomousvision/mip-splatting.git $BUILD_DIR/mip-splatting \
    --depth 1 --quiet
cd $BUILD_DIR/mip-splatting/submodules/diff-gaussian-rasterization
$PIP install . --no-build-isolation -q
echo "  diff-gaussian-rasterization: DONE"

# ── 2. Rebuild nvdiffrast ──────────────────────────────────────────────────────
echo ""
echo "=== Rebuilding nvdiffrast for sm_${SM} ==="
git clone https://github.com/NVlabs/nvdiffrast.git $BUILD_DIR/nvdiffrast \
    --depth 1 --quiet
cd $BUILD_DIR/nvdiffrast
$PIP install . --no-build-isolation -q
echo "  nvdiffrast: DONE"

# ── 3. Quick smoke test ────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
cd /net/projects/ranalab/rajhansini/TRELLIS
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test diff_gaussian_rasterization
try:
    import diff_gaussian_rasterization
    print("diff_gaussian_rasterization: import OK")
except Exception as e:
    print(f"diff_gaussian_rasterization: FAILED -> {e}")

# Test nvdiffrast
try:
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
    print("nvdiffrast RasterizeCudaContext: OK")
    del glctx
except Exception as e:
    print(f"nvdiffrast RasterizeCudaContext: FAILED -> {e}")
PYEOF

echo ""
echo "=== Build complete. Now run submit_phase8.sh ==="
