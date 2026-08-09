#!/bin/bash
# Rebuild nvdiffrast + diff-gaussian-rasterization for h002 (H200, sm_90)
set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip

echo "=== GPU info ==="
nvidia-smi -L

echo "=== Verifying CUDA from trellis env ==="
$PY -c "import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(f'GPU: {p.name}  sm_{p.major}{p.minor}  {p.total_memory//1024**3}GB')"

export TORCH_CUDA_ARCH_LIST="9.0"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

BUILD_DIR=/tmp/renderer_build_h002_$$
mkdir -p "$BUILD_DIR"
echo "Build dir: $BUILD_DIR"

# ── 1. nvdiffrast ─────────────────────────────────────────────────────────────
echo ""
echo "=== Rebuilding nvdiffrast for sm_90 ==="
git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast" --depth 1 --quiet
cd "$BUILD_DIR/nvdiffrast"
$PIP install . --no-build-isolation --no-cache-dir -q
echo "  nvdiffrast: DONE"

# ── 2. diff-gaussian-rasterization ────────────────────────────────────────────
echo ""
echo "=== Rebuilding diff-gaussian-rasterization for sm_90 ==="
git clone https://github.com/autonomousvision/mip-splatting.git "$BUILD_DIR/mip-splatting" --depth 1 --quiet
cd "$BUILD_DIR/mip-splatting/submodules/diff-gaussian-rasterization"
$PIP install . --no-build-isolation --no-cache-dir -q
echo "  diff-gaussian-rasterization: DONE"

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
cd /net/projects/ranalab/rajhansini/TRELLIS
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
try:
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
    print("nvdiffrast RasterizeCudaContext: OK")
    del glctx
except Exception as e:
    print(f"nvdiffrast: FAILED -> {e}")
try:
    import diff_gaussian_rasterization
    print("diff_gaussian_rasterization: import OK")
except Exception as e:
    print(f"diff_gaussian_rasterization: FAILED -> {e}")
PYEOF

rm -rf "$BUILD_DIR"
echo ""
echo "=== Build complete for h002 (sm_90) ==="
