#!/bin/bash
# Rebuild nvdiffrast for current GPU arch (L40S = sm_89 on q001/q002)
set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip

for CUDA_CANDIDATE in /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.1 /usr/local/cuda; do
    if [ -f "$CUDA_CANDIDATE/bin/nvcc" ]; then
        export CUDA_HOME=$CUDA_CANDIDATE
        break
    fi
done
if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: nvcc not found"; exit 1
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "CUDA_HOME=${CUDA_HOME}"

echo "=== GPU info ==="
nvidia-smi -L

SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
echo "Detected SM: sm_${SM}"
export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

BUILD_DIR=/tmp/nvdiffrast_rebuild_$$
mkdir -p $BUILD_DIR

echo ""
echo "=== Cloning nvdiffrast ==="
git clone https://github.com/NVlabs/nvdiffrast.git $BUILD_DIR/nvdiffrast --depth 1
cd $BUILD_DIR/nvdiffrast

echo ""
echo "=== Building for sm_${SM} ==="
$PIP install . --no-build-isolation

echo ""
echo "=== Smoke test ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
print("nvdiffrast RasterizeCudaContext: OK")
del glctx
PYEOF

echo ""
echo "=== nvdiffrast rebuild DONE for sm_${SM} ==="
rm -rf $BUILD_DIR
