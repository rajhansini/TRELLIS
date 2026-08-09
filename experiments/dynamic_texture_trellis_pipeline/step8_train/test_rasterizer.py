"""Test the diff_gaussian_rasterization with snapshot data."""
import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

SNAP = '/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train/snapshot_fw.dump'
args = torch.load(SNAP, map_location='cuda', weights_only=False)
(bg, means3D, colors_precomp, opacities, scales, rotations, scale_modifier,
 cov3Ds_precomp, viewmatrix, projmatrix, tanfovx, tanfovy, kernel_size,
 subpixel_offset, image_height, image_width, sh, sh_degree, campos,
 prefiltered, debug_flag) = args

print(f"N total Gaussians: {means3D.shape[0]}")
print(f"viewmatrix:\n{viewmatrix}")

def try_render(N):
    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height), image_width=int(image_width),
        tanfovx=float(tanfovx), tanfovy=float(tanfovy),
        kernel_size=float(kernel_size),
        subpixel_offset=subpixel_offset,
        bg=bg, scale_modifier=float(scale_modifier),
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=int(sh_degree), campos=campos,
        prefiltered=bool(prefiltered), debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros(N, 3, device='cuda', requires_grad=True)
    try:
        color, radii = rasterizer(
            means3D=means3D[:N].contiguous(),
            means2D=means2D,
            shs=sh[:N].contiguous(),
            colors_precomp=None,
            opacities=opacities[:N].contiguous(),
            scales=None, rotations=None,
            cov3D_precomp=cov3Ds_precomp[:N].contiguous(),
        )
        visible = (radii > 0).sum().item()
        max_r = radii.max().item()
        print(f"  N={N:>8}: SUCCESS  visible={visible}, max_radius={max_r}")
        return True
    except torch.cuda.OutOfMemoryError as e:
        print(f"  N={N:>8}: OOM — {str(e)[:120]}")
        return False
    except Exception as e:
        print(f"  N={N:>8}: ERROR — {e}")
        return False

for N in [1, 100, 10000]:
    ok = try_render(N)
    torch.cuda.empty_cache()

print("\n--- Testing with SYNTHETIC data (random positions, zero cov) ---")
for N in [1, 100, 10000]:
    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height), image_width=int(image_width),
        tanfovx=float(tanfovx), tanfovy=float(tanfovy),
        kernel_size=float(kernel_size),
        subpixel_offset=subpixel_offset,
        bg=bg, scale_modifier=float(scale_modifier),
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=int(sh_degree), campos=campos,
        prefiltered=bool(prefiltered), debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    torch.manual_seed(42)
    syn_xyz = torch.rand(N, 3, device='cuda') - 0.5  # positions in [-0.5, 0.5]
    syn_cov = torch.zeros(N, 6, device='cuda')
    syn_sh  = torch.ones(N, 1, 3, device='cuda') * 0.5
    syn_opa = torch.ones(N, 1, device='cuda') * 0.5
    syn_m2d = torch.zeros(N, 3, device='cuda', requires_grad=True)
    try:
        color, radii = rasterizer(
            means3D=syn_xyz, means2D=syn_m2d,
            shs=syn_sh, colors_precomp=None,
            opacities=syn_opa, scales=None, rotations=None,
            cov3D_precomp=syn_cov,
        )
        print(f"  SYNTHETIC N={N}: SUCCESS  visible={(radii>0).sum()}, max_r={radii.max()}")
    except Exception as e:
        print(f"  SYNTHETIC N={N}: FAILED — {str(e)[:100]}")
    torch.cuda.empty_cache()

print("\n--- Testing with IDENTITY viewmatrix ---")
I4 = torch.eye(4, device='cuda')
I4[2, 3] = 2.0  # translate camera back
for N in [100]:
    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height), image_width=int(image_width),
        tanfovx=float(tanfovx), tanfovy=float(tanfovy),
        kernel_size=float(kernel_size),
        subpixel_offset=subpixel_offset,
        bg=bg, scale_modifier=float(scale_modifier),
        viewmatrix=I4, projmatrix=projmatrix,
        sh_degree=int(sh_degree), campos=campos,
        prefiltered=bool(prefiltered), debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    syn_xyz = torch.rand(N, 3, device='cuda') - 0.5
    syn_cov = torch.zeros(N, 6, device='cuda')
    syn_sh  = torch.ones(N, 1, 3, device='cuda') * 0.5
    syn_opa = torch.ones(N, 1, device='cuda') * 0.5
    syn_m2d = torch.zeros(N, 3, device='cuda', requires_grad=True)
    try:
        color, radii = rasterizer(
            means3D=syn_xyz, means2D=syn_m2d,
            shs=syn_sh, colors_precomp=None,
            opacities=syn_opa, scales=None, rotations=None,
            cov3D_precomp=syn_cov,
        )
        print(f"  IDENTITY view N={N}: SUCCESS  visible={(radii>0).sum()}, max_r={radii.max()}")
    except Exception as e:
        print(f"  IDENTITY view N={N}: FAILED — {str(e)[:100]}")
    torch.cuda.empty_cache()
