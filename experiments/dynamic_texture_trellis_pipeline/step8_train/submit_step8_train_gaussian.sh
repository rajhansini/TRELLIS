#!/bin/bash
#SBATCH --job-name=step8_train_gs
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

mkdir -p ../results/lora_ckpts_gs

echo "=== GPU info ==="
nvidia-smi -L || true

SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
echo "GPU SM: sm_${SM}"

echo "=== Checking diff-gaussian-rasterization compatibility ==="
set +e
SPCONV_ALGO=native ATTN_BACKEND=xformers $PY - <<'PYEOF'
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
N = 4
bg = torch.ones(3, device='cuda')
vm = torch.eye(4, device='cuda'); vm[2,3] = 2.0
fov = math.radians(40); tanf = math.tan(fov/2)
proj = torch.zeros(4,4,device='cuda')
proj[0,0]=2.747; proj[1,1]=-2.747; proj[2,2]=-2.0
proj[2,3]=-1.0; proj[3,2]=2.4; proj[3,3]=2.0
rs = GaussianRasterizationSettings(
    image_height=64, image_width=64,
    tanfovx=tanf, tanfovy=tanf, kernel_size=0.1,
    subpixel_offset=torch.zeros(64,64,2,device='cuda'),
    bg=bg, scale_modifier=1.0, viewmatrix=vm, projmatrix=proj,
    sh_degree=0, campos=torch.tensor([0.,0.,0.],device='cuda'),
    prefiltered=False, debug=False
)
r = GaussianRasterizer(rs)
xyz = torch.rand(N,3,device='cuda') - 0.5
m2d = torch.zeros(N,3,device='cuda',requires_grad=True)
opa = torch.ones(N,1,device='cuda')*0.5
sh_ = torch.ones(N,1,3,device='cuda')*0.5
cov = torch.zeros(N,6,device='cuda')
color, radii = r(means3D=xyz,means2D=m2d,shs=sh_,colors_precomp=None,
                 opacities=opa,scales=None,rotations=None,cov3D_precomp=cov)
print("diff_gaussian_rasterization OK")
PYEOF
GS_OK=$?
set -e

if [ "$GS_OK" -ne 0 ]; then
    echo "=== Rebuilding diff-gaussian-rasterization for sm_${SM} ==="
    export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
    BUILD_DIR=$(mktemp -d)
    git clone https://github.com/autonomousvision/mip-splatting.git $BUILD_DIR/mip-splatting --depth 1 --quiet
    cd $BUILD_DIR/mip-splatting/submodules/diff-gaussian-rasterization
    $PIP install . --no-build-isolation -q
    cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train
    rm -rf $BUILD_DIR
    echo "=== Rebuild done ==="
fi

echo "=== Step 8 Gaussian Training ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u train_gaussian.py 2>&1 | tee ../results/train_step8_gs.log
TRAIN_EXIT=${PIPESTATUS[0]}
if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== train_gaussian.py FAILED (exit $TRAIN_EXIT) ==="
    exit $TRAIN_EXIT
fi

echo "=== Training done, starting inference ==="
SPCONV_ALGO=native ATTN_BACKEND=xformers \
    $PY -u infer_gaussian.py 2>&1 | tee ../results/infer_gaussian.log

echo "=== Done ==="
