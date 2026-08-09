#!/bin/bash
#SBATCH --job-name=build_gs_raster
#SBATCH --partition=threedle-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --nodelist=c001
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/build_gs_%j.log

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
PIP=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/pip

echo "=== GPU info ==="
nvidia-smi -L

SM=$($PY -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
echo "Detected SM: sm_${SM}"
export TORCH_CUDA_ARCH_LIST="${SM:0:1}.${SM:1:1}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

BUILD_DIR=/tmp/gs_build_$$
mkdir -p $BUILD_DIR

echo "=== Cloning mip-splatting ==="
git clone https://github.com/autonomousvision/mip-splatting.git $BUILD_DIR/mip-splatting \
    --depth 1 --quiet
cd $BUILD_DIR/mip-splatting/submodules/diff-gaussian-rasterization
echo "=== Installing diff-gaussian-rasterization for sm_${SM} ==="
$PIP install . --no-build-isolation
echo "=== Done ==="

echo "=== Smoke test ==="
$PY -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
N = 10
bg = torch.ones(3, device='cuda')
vm = torch.eye(4, device='cuda'); vm[2,3] = 2.0
fov = math.radians(40)
tanf = math.tan(fov/2)
proj = torch.zeros(4,4,device='cuda')
proj[0,0]=2.747; proj[1,1]=-2.747; proj[2,2]=-2.0; proj[2,3]=-1.0; proj[3,2]=2.4; proj[3,3]=2.0
rs = GaussianRasterizationSettings(
    image_height=256, image_width=256,
    tanfovx=tanf, tanfovy=tanf, kernel_size=0.1,
    subpixel_offset=torch.zeros(256,256,2,device='cuda'),
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
color, radii = r(means3D=xyz, means2D=m2d, shs=sh_, colors_precomp=None,
                 opacities=opa, scales=None, rotations=None, cov3D_precomp=cov)
print(f'Smoke test PASSED: color={color.shape}, visible={(radii>0).sum().item()}')
"

echo "=== Rebuild complete ==="
