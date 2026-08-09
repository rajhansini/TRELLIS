#!/bin/bash
#SBATCH --job-name=infer_step8
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/pip

echo "=== GPU info ==="
nvidia-smi -L || true

echo "=== Checking nvdiffrast compatibility with this GPU ==="
set +e
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
print("nvdiffrast OK on this GPU")
PYEOF
NVDIFFRAST_OK=$?
set -e

if [ "$NVDIFFRAST_OK" -ne 0 ]; then
    echo "=== nvdiffrast incompatible with this GPU arch — rebuilding ==="
    SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
    echo "Detected SM version: sm_${SM}"
    export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"

    BUILD_DIR=$(mktemp -d)
    git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast" --depth 1 --quiet
    cd "$BUILD_DIR/nvdiffrast"
    $PIP install . --no-build-isolation -q
    cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
    rm -rf "$BUILD_DIR"
fi

echo "=== Running inference ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY -u infer.py
