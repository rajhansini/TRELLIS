"""
Step 8 — SLaT Decode + nvdiffrast Render.

BACKWARD COMPATIBLE: original train.py untouched.
This module is imported by train_v2.py and any future inference scripts.

Exports:
  RENDER_RES   : int         — 518 (native DINOv2 resolution)
  INTRINSICS   : Tensor(3,3) — normalized camera intrinsics (fov=40°)
  EXTRINSICS   : Tensor(4,4) — confirmed front-view extrinsics for teapot
  SLAT_MEAN    : Tensor(8,)
  SLAT_STD     : Tensor(8,)
  make_renderer()             — build MeshRenderer with correct options
  normalize_slat(x0)          — un-normalize SLaT feats from flow space
  decode_and_render(pipeline, slat, renderer, diag=False)
                              — returns (color Tensor(3,H,W), slat_leaf_feats)
  render_slat(pipeline, slat, renderer, diag=False)
                              — convenience: normalize → decode → render → (3,H,W)

Camera notes (confirmed in verify_step7):
  cam_x=(1,0,0)  cam_y=(0,0,-1)  cam_z=(0,1,0)  translation=(0,0,2)
  TRELLIS objects at ~[-0.5,0.5]^3, camera at z=2 looking down -Y world axis
  nvdiffrast Y convention flips cam_y row vs TRELLIS Gaussian renderer
"""

import math
import torch
import torch.nn as nn
from trellis.modules import sparse as sp
from trellis.renderers import MeshRenderer

# ── Constants ─────────────────────────────────────────────────────────────────

RENDER_RES = 518

_fx_n = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))

# Confirmed correct for MeshRenderer teapot front view.
# cam_x=(1,0,0), cam_y=(0,0,-1), cam_z=(0,1,0), t=(0,0,2)
EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32)

INTRINSICS = torch.tensor(
    [[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
    dtype=torch.float32,
)

# SLaT channel normalization constants (from pipeline.json)
SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,  1.2039363384246826,
], dtype=torch.float32)

SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338,
], dtype=torch.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_renderer(device: torch.device = None) -> MeshRenderer:
    """Build MeshRenderer with correct options for TRELLIS teapot front view."""
    opts = {'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    renderer = MeshRenderer(rendering_options=opts)
    print(f'[S8] MeshRenderer: res={RENDER_RES}  near=0.5  far=3.0  ssaa=1')
    return renderer


def normalize_slat(x0: sp.SparseTensor) -> sp.SparseTensor:
    """Un-normalize SLaT features from flow (unit-normal) space to mesh-decoder space."""
    dev  = x0.feats.device
    std  = SLAT_STD.to(dev)
    mean = SLAT_MEAN.to(dev)
    out  = x0.replace(x0.feats * std + mean)
    return out


# ── Diagnostic state (printed once per process) ───────────────────────────────

_DIAG_DONE  = False
_DIAG_HOOKS = []   # keep handles to prevent GC


def reset_diag():
    """Call between runs in a notebook/script to re-enable one-shot diagnostics."""
    global _DIAG_DONE
    _DIAG_DONE = False


def _gfn(t):
    if t is None:
        return 'None'
    fn = type(t.grad_fn).__name__ if t.grad_fn else 'None'
    return f'req={t.requires_grad} fn={fn} shape={tuple(t.shape)} dtype={t.dtype}'


# ── Core decode + render ──────────────────────────────────────────────────────

def decode_and_render(
    pipeline,
    slat: sp.SparseTensor,
    renderer: MeshRenderer,
    diag: bool = False,
    device: torch.device = None,
) -> tuple:
    """
    Decode SLaT sparse tensor → mesh → nvdiffrast color image.

    IMPORTANT: slat.feats must already be in mesh-decoder space
    (i.e. after normalize_slat). Do NOT pass flow-space feats.

    Args:
      pipeline : TrellisImageTo3DPipeline  (decoder on same device as slat)
      slat     : SparseTensor with feats in mesh-decoder space
      renderer : MeshRenderer from make_renderer()
      diag     : print one-shot gradient-chain diagnostics (auto-disabled after first call)
      device   : torch.device; inferred from slat.feats if None

    Returns:
      color           : Tensor (3, RENDER_RES, RENDER_RES) — composited over white
      slat_leaf_feats : Tensor (N_vox, ch) with requires_grad=True — use for gradient injection
    """
    global _DIAG_DONE, _DIAG_HOOKS

    if device is None:
        device = slat.feats.device

    ext = EXTRINSICS.to(device)
    intr = INTRINSICS.to(device)

    do_diag = diag and not _DIAG_DONE

    # ── Leaf node for gradient injection ─────────────────────────────────────
    slat_leaf_feats = slat.feats.detach().requires_grad_(True)
    slat_for_decode = sp.SparseTensor(feats=slat_leaf_feats, coords=slat.coords)

    if do_diag:
        print(f'  [S8|G0] slat_leaf_feats : {_gfn(slat_leaf_feats)}')

    # ── Decode SLaT → mesh ────────────────────────────────────────────────────
    decoded = pipeline.decode_slat(slat_for_decode, ['mesh'])
    mesh    = decoded['mesh'][0]

    if do_diag:
        print(f'  [S8|M1] mesh.vertices    : {_gfn(mesh.vertices)}')
        print(f'  [S8|M2] mesh.vertex_attrs: {_gfn(mesh.vertex_attrs)}')
        v = mesh.vertices
        print(f'  [S8|M2] vertex XYZ range : '
              f'x=[{v[:,0].min():.3f},{v[:,0].max():.3f}] '
              f'y=[{v[:,1].min():.3f},{v[:,1].max():.3f}] '
              f'z=[{v[:,2].min():.3f},{v[:,2].max():.3f}]')

    # ── Render ────────────────────────────────────────────────────────────────
    result = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    mask   = result['mask'].unsqueeze(0)              # (1, H, W)
    color  = result['color'] * mask + (1.0 - mask)   # composite over white (3, H, W)

    if do_diag:
        print(f'  [S8|M3] rendered color   : {_gfn(color)}')
        cov = mask.float().mean().item() * 100
        print(f'  [S8|M3] mask coverage    : {cov:.1f}%')
        print(f'  [S8|M3] color range      : [{color.min():.4f}, {color.max():.4f}]')

        def _bhook(name):
            def _h(g):
                mx = g.abs().max().item()
                print(f'  [S8|BWD] {name}: dtype={g.dtype} max={mx:.3e} '
                      f'mean={g.abs().mean().item():.3e} '
                      f'{"  <<ZERO!" if mx == 0.0 else "  ok"}')
            return _h

        _DIAG_HOOKS.clear()
        _DIAG_HOOKS.append(color.register_hook(_bhook('color')))
        _DIAG_HOOKS.append(mesh.vertex_attrs.register_hook(_bhook('vertex_attrs')))
        _DIAG_HOOKS.append(slat_leaf_feats.register_hook(_bhook('slat_leaf_feats')))

        _DIAG_DONE = True

    return color, slat_leaf_feats


def render_slat(
    pipeline,
    slat: sp.SparseTensor,
    renderer: MeshRenderer,
    diag: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convenience: normalize_slat → decode_and_render → return color only.
    For training use decode_and_render directly (need slat_leaf_feats for grad).

    Returns color : Tensor (3, RENDER_RES, RENDER_RES)
    """
    slat_norm = normalize_slat(slat)
    color, _ = decode_and_render(pipeline, slat_norm, renderer, diag=diag, device=device)
    return color
