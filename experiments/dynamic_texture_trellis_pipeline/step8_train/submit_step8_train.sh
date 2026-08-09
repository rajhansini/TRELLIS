#!/bin/bash
#SBATCH --job-name=step8_train
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/slurm_%j.err

set -e
cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/pip

mkdir -p ../results/lora_ckpts

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
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

    BUILD_DIR=$(mktemp -d)
    git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast" --depth 1 --quiet
    cd "$BUILD_DIR/nvdiffrast"
    $PIP install . --no-build-isolation -q
    cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
    rm -rf "$BUILD_DIR"

    echo "=== Re-checking nvdiffrast ==="
    SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
print("nvdiffrast OK after rebuild")
PYEOF
fi

echo "=== Step 8 Training ==="
# -u: unbuffered stdout/stderr so train.log is updated in real-time
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u train.py 2>&1 | tee ../results/train.log
TRAIN_EXIT=${PIPESTATUS[0]}
if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== train.py FAILED (exit $TRAIN_EXIT) — aborting before inference ==="
    exit $TRAIN_EXIT
fi

echo "=== Training done, starting inference ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u infer.py 2>&1 | tee ../results/infer.log
INFER_EXIT=${PIPESTATUS[0]}
if [ "$INFER_EXIT" -ne 0 ]; then
    echo "=== infer.py FAILED (exit $INFER_EXIT) ==="
    exit $INFER_EXIT
fi

echo "=== Done ==="
