#!/usr/bin/env python3
"""
Run 0 — Baseline: raw TRELLIS per frame, no MCFM, no LoRA.

Single-frame DINOv2 conditioning only. Fixed structure seed + fixed denoising
seed for consistency with all later runs. Evaluates masked PSNR and LPIPS
on held-out frames. Every later run is measured against these numbers.

Output: lora_experiments/run00_baseline/
  renders/frame_XXXX.png   — rendered output for each held-out frame
  metrics.json             — per-frame + aggregate PSNR and LPIPS
  metrics_summary.txt      — human-readable printout
"""

import sys
import os
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
TRELLIS_ROOT = Path('/net/projects/ranalab/rajhansini/TRELLIS')

_OUT_DIR_EARLY = TRELLIS_ROOT / 'experiments' / 'lora_experiments' / 'run00_baseline'
_OUT_DIR_EARLY.mkdir(parents=True, exist_ok=True)

class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(_OUT_DIR_EARLY / 'train.log')
sys.stderr = sys.stdout
sys.path.insert(0, str(TRELLIS_ROOT))

sys.path.insert(0, str(TRELLIS_ROOT / 'experiments/dynamic_texture_trellis_pipeline/step8_decode_render'))
sys.path.insert(0, str(TRELLIS_ROOT / 'experiments/dynamic_texture_trellis_pipeline'))

from decode_render import (
    RENDER_RES, make_renderer, normalize_slat, decode_and_render,
)

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
import torchvision.transforms as T

import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('SPCONV_ALGO',  'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

# ── Constants ─────────────────────────────────────────────────────────────────
GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
N_FRAMES    = 150
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
RESCALE_T   = 3.0

# Held-out eval set: 15 frames, evenly spaced across the video
EVAL_FRAMES = list(range(1, 151))        # all 150 frames

OUT_DIR = Path(__file__).parent / 'run00_baseline'
RENDER_DIR = OUT_DIR / 'renders'

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_mask() -> torch.Tensor:
    """Return boolean mask (H, W) on DEVICE, True = object pixel."""
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)   # (H, W)


def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((518, 518), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    x   = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)   # (n_tokens, feat_dim)


def load_gt_tensor(frame_idx: int) -> torch.Tensor:
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR on masked pixels only. pred/gt: (3,H,W) in [0,1]. mask: (H,W) bool."""
    diff   = (pred - gt)[:, mask]          # (3, n_mask)
    mse    = diff.pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def render_frame(flow_model, pipeline, tokens, coords, noise_feats, renderer):
    """Denoise a single frame with raw TRELLIS (no MCFM, no LoRA)."""
    ns      = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    cond_gl = tokens.unsqueeze(0).to(DEVICE)     # (1, n_tokens, feat_dim)

    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(ns, t_ten, cond_gl)
            ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)

    slat  = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    return color.detach().clamp(0, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 68)
    print('Run 00 — Baseline: raw TRELLIS, no MCFM, no LoRA')
    print('=' * 68)
    print(f'  GT frames : {GT_FRAMES_DIR}')
    print(f'  eval set  : {EVAL_FRAMES}  ({len(EVAL_FRAMES)} frames)')
    print(f'  seeds     : STRUCT={STRUCT_SEED}  DENOISE={FIXED_SEED}')
    print(f'  output    : {OUT_DIR}')

    # ── Sanity check GT frames ─────────────────────────────────────────────
    missing = [f for f in EVAL_FRAMES
               if not (GT_FRAMES_DIR / f'frame_{f:04d}.png').exists()]
    if missing:
        raise FileNotFoundError(f'Missing GT frames: {missing}')
    print('\n[GT] All eval frames found.')

    # ── Load mask ─────────────────────────────────────────────────────────
    mask = load_mask()
    n_mask = mask.sum().item()
    print(f'[MASK] {MASK_PATH.name}: {n_mask} masked pixels '
          f'({100*n_mask/(RENDER_RES**2):.1f}% of frame)')

    # ── Load LPIPS ────────────────────────────────────────────────────────
    print('\n[LPIPS] Loading AlexNet perceptual loss...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    lpips_fn.eval()

    # ── Load pipeline ─────────────────────────────────────────────────────
    print(f'\n[TRELLIS] Loading pipeline from {PRETRAINED}...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Voxel structure (fixed seed) ───────────────────────────────────────
    # Sample structure BEFORE offloading: sample_sparse_structure needs the
    # sparse-structure models on GPU, and pipeline.device is derived from the
    # first model's device — offloading it first makes get_cond move the image
    # to CPU while DINOv2 weights stay on CUDA (device mismatch crash).
    print(f'\n[STRUCT] Sampling structure with seed={STRUCT_SEED}...')
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}')
    del cond_struct
    gc.collect(); torch.cuda.empty_cache()

    # Offload now-unused models to save VRAM (after structure sampling)
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try:
                pipeline.models[name].cpu()
            except Exception:
                pass
    torch.cuda.empty_cache()

    # ── Fixed denoising noise ──────────────────────────────────────────────
    print(f'\n[NOISE] Fixed denoising noise with seed={FIXED_SEED}')
    torch.manual_seed(FIXED_SEED)
    noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # ── Encode all frames with DINOv2 ──────────────────────────────────────
    print(f'\n[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu()
    gc.collect(); torch.cuda.empty_cache()

    # ── Renderer ──────────────────────────────────────────────────────────
    renderer = make_renderer()

    # ── Evaluate held-out frames ───────────────────────────────────────────
    print(f'\n[EVAL] Running {len(EVAL_FRAMES)} held-out frames...\n')
    results = []

    for fi in EVAL_FRAMES:
        print(f'  frame {fi:03d}/', end='', flush=True)

        tokens = raw_tokens[fi].to(DEVICE)
        render = render_frame(flow_model, pipeline, tokens, coords, noise_feats, renderer)

        gt     = load_gt_tensor(fi)

        # Masked PSNR
        psnr = masked_psnr(render, gt, mask)

        # LPIPS — expects (1,3,H,W) in [-1,1]
        r_lp = render.unsqueeze(0) * 2 - 1
        g_lp = gt.unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(r_lp, g_lp).item()

        print(f'  PSNR={psnr:.3f} dB   LPIPS={lp:.4f}')

        # Save render
        out_png = RENDER_DIR / f'frame_{fi:04d}.png'
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_png)

        results.append({'frame': fi, 'psnr': psnr, 'lpips': lp})

        del render, gt, tokens
        gc.collect(); torch.cuda.empty_cache()

    # ── Aggregate ─────────────────────────────────────────────────────────
    psnrs  = [r['psnr']  for r in results]
    lpipss = [r['lpips'] for r in results]

    summary = {
        'run':          'run00_baseline',
        'description':  'Raw TRELLIS, no MCFM, no LoRA. Fixed seeds.',
        'seeds':        {'struct': STRUCT_SEED, 'denoise': FIXED_SEED},
        'eval_frames':  EVAL_FRAMES,
        'n_eval':       len(EVAL_FRAMES),
        'mask_pixels':  int(n_mask),
        'psnr_mean':    float(np.mean(psnrs)),
        'psnr_std':     float(np.std(psnrs)),
        'psnr_min':     float(np.min(psnrs)),
        'psnr_max':     float(np.max(psnrs)),
        'lpips_mean':   float(np.mean(lpipss)),
        'lpips_std':    float(np.std(lpipss)),
        'per_frame':    results,
    }

    metrics_path = OUT_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print('\n' + '=' * 68)
    print('Run 00 — RESULTS')
    print('=' * 68)
    print(f'  Masked PSNR : {summary["psnr_mean"]:.3f} ± {summary["psnr_std"]:.3f} dB')
    print(f'  LPIPS       : {summary["lpips_mean"]:.4f} ± {summary["lpips_std"]:.4f}')
    print(f'  eval frames : {len(EVAL_FRAMES)} frames (every 10th)')
    print(f'  metrics     → {metrics_path}')
    print(f'  renders     → {RENDER_DIR}')
    print('=' * 68)

    txt = OUT_DIR / 'metrics_summary.txt'
    with open(txt, 'w') as f:
        f.write(f'Run 00 — Baseline: raw TRELLIS, no MCFM, no LoRA\n')
        f.write(f'seeds: STRUCT={STRUCT_SEED}, DENOISE={FIXED_SEED}\n')
        f.write(f'eval:  {len(EVAL_FRAMES)} held-out frames (every 10th, frames 10..150)\n\n')
        f.write(f'Masked PSNR : {summary["psnr_mean"]:.3f} ± {summary["psnr_std"]:.3f} dB\n')
        f.write(f'LPIPS       : {summary["lpips_mean"]:.4f} ± {summary["lpips_std"]:.4f}\n\n')
        f.write('Per-frame:\n')
        for r in results:
            f.write(f'  frame {r["frame"]:03d}:  PSNR={r["psnr"]:.3f} dB  LPIPS={r["lpips"]:.4f}\n')


if __name__ == '__main__':
    main()
