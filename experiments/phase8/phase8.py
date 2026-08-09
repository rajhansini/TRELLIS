"""
Phase 8 — Temporal Attention Smoothing in SLaT Feature Space
=============================================================
Correct implementation: temporal smoothing happens BEFORE decoding,
inside the Gaussian Latent (SLaT) space (N_voxels × 8), so TRELLIS's
decoder sees temporally-aware features when generating each frame.

For each frame t:
  1. Build window of SLaT features [z_{t-k}, ..., z_t, ..., z_{t+k}]
  2. Apply temporal attention → smoothed z̃_t  (N_vox, 8)
  3. Decode z̃_t via pipeline.decode_slat → Gaussians
  4. Render, compare to GT

4 modes:

  avg (V1):
    K̂ = V̂ = mean over window frames        (N, 8)
    z̃_t = softmax(z_t @ K̂ᵀ / √8) @ V̂      spatial attention on averaged K,V

  temporal_spatial (V2):
    Stage 1 — per-voxel temporal blend:
      scores_t[v, t'] = z_t[v] · z_{t'}[v] / √8     (N, W)
      α boost on current frame t before softmax
      K̂_v = V̂_v = Σ_t' softmax(scores_t)[v,t'] · z_{t'}[v]
    Stage 2 — spatial attention on blended K̂, V̂:
      z̃_t = softmax(z_t @ K̂ᵀ / √8) @ V̂

  spatial_temporal (V2b):
    Stage 1 — spatial self-attention within frame t:
      spatial_out = softmax(z_t @ z_tᵀ / √8) @ z_t
    Stage 2 — per-voxel temporal blend using spatial_out as query:
      scores_t[v, t'] = spatial_out[v] · z_{t'}[v] / √8     (N, W)
      α boost on current frame t before softmax
      z̃_t[v] = Σ_t' softmax(scores_t)[v,t'] · z_{t'}[v]

  joint (V3):
    Pool all window frames: K_all = [z_{t-k}; ...; z_{t+k}]  (W·N, 8)
    α boost on current frame block
    z̃_t = softmax(z_t @ K_allᵀ / √8) @ K_all    single joint attention

Knobs:
  --k        temporal window half-size (default 1)
  --alpha    current-frame key boost in temporal attention (default 1.0)
  --mode     avg | temporal_spatial | spatial_temporal | joint | all

Input:
  experiments/results/phase7/slat_cache.npz  —  (150, 6281, 8) SLaT features
  microsoft/TRELLIS-image-large              —  pipeline (for decode_slat)

Output: experiments/results/phase8/<mode>_k<k>/
  frame_NNN.png, psnr.csv, render.mp4, comparison.mp4

Run:
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase8/phase8.py --k 1 --mode all
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import csv
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils
from trellis.renderers import GaussianRenderer

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PHASE7_OUT   = REPO_ROOT / 'experiments' / 'results' / 'phase7'
OUT_DIR      = REPO_ROOT / 'experiments' / 'results' / 'phase8'
GT_VIDEO_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                    '/outputs/teapot_lava_kling_premium'
                    '/teapot_lava_kling_premium_front/all_frames_150')
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
PRETRAINED   = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
RENDER_RES = 512
FEAT_DIM   = 8


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_slat_cache():
    """Load SLaT features and coords from phase7 cache."""
    path = PHASE7_OUT / 'slat_cache.npz'
    if not path.exists():
        raise FileNotFoundError(
            f'SLaT cache not found at {path}. Run phase7 first to generate it.')
    data   = np.load(path)
    slats  = data['slats'].astype(np.float32)   # (T, N_vox, 8)
    coords = data['coords'].astype(np.int32)     # (N_vox, 3) without batch col
    T, N, D = slats.shape
    print(f'SLaT cache: T={T}  N_vox={N}  feat_dim={D}')
    print(f'  feats: min={slats.min():.4f}  max={slats.max():.4f}'
          f'  nan={np.isnan(slats).sum()}  inf={np.isinf(slats).sum()}')
    # Add batch column of zeros → (N_vox, 4)
    batch        = np.zeros((N, 1), dtype=np.int32)
    coords_batch = np.concatenate([batch, coords], axis=1)
    return slats, coords_batch


def build_camera():
    import utils3d.torch as u3d
    fov  = torch.deg2rad(torch.tensor(40.)).cuda()
    eye  = torch.tensor([0., 0., 2.]).cuda()
    tgt  = torch.zeros(3).cuda()
    up   = torch.tensor([0., 1., 0.]).cuda()
    extr = u3d.extrinsics_look_at(eye, tgt, up)
    intr = u3d.intrinsics_from_fov_xy(fov, fov)
    return [extr], [intr]


def load_gt(frame_idx, device):
    path = GT_VIDEO_DIR / f'frame_{frame_idx:04d}.png'
    img  = Image.open(path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.).permute(2, 0, 1).to(device)


def to_uint8(t):
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def psnr(pred, gt):
    mse = float(F.mse_loss(pred.detach(), gt.detach()).item())
    return 10 * math.log10(1.0 / (mse + 1e-10))



_SCALE_WARN_DONE = False

def decode_and_render(pipeline, feats_t, coords_t, extr, intr):
    """Decode SLaT features → Gaussians → render. Returns (H,W,3) uint8."""
    global _SCALE_WARN_DONE
    slat = sp.SparseTensor(feats=feats_t, coords=coords_t)
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['gaussian'])
    g = decoded['gaussian'][0]
    # Clamp log-scales: get_scaling = exp(_scaling + scale_bias).
    # Values outside [-6, 2] cause the rasterizer to try absurd tile allocations.
    raw_max = g._scaling.max().item()
    g._scaling = g._scaling.clamp(-6.0, 2.0)
    if not _SCALE_WARN_DONE and raw_max > 2.0:
        print(f'[decode_and_render] _scaling clamped from max={raw_max:.2f} → 2.0')
        _SCALE_WARN_DONE = True
    frames = render_utils.render_frames(
        g, extr, intr,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return frames['color'][0]  # (H, W, 3) uint8


def make_video(frames_dir, pattern, out_path, fps=15):
    try:
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
            '-i', str(frames_dir / pattern),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
        ], check=True, capture_output=True)
        print(f'Saved: {out_path}')
    except Exception as e:
        print(f'Video failed: {e}')


def make_comparison_video(gt_dir, trellis_dir, smoothed_dir, out_path):
    """3-panel: GT | TRELLIS (flicker) | phase8 method."""
    try:
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '15',
            '-i', str(gt_dir      / 'frame_%03d.png'),
            '-i', str(trellis_dir / 'frame_%03d.png'),
            '-i', str(smoothed_dir / 'frame_%03d.png'),
            '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
        ], check=True, capture_output=True)
        print(f'Saved: {out_path}')
    except Exception as e:
        print(f'Comparison video failed: {e}')


def save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'Saved: {path}')


def prepare_trellis_renders(out_dir):
    """Copy per-frame TRELLIS baseline renders (flicker baseline) for comparison panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(1, N_FRAMES + 1):
        dst = out_dir / f'frame_{t:03d}.png'
        if dst.exists():
            continue
        src = FRAMES_150_DIR / f'frame_{t:04d}' / 'renders' / 'front.png'
        img = Image.open(src).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        img.save(dst)
    print(f'TRELLIS baseline renders ready: {out_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# V1: Average K,V across window → spatial attention
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_attn(q, kv):
    """Spatial attention with cosine similarity — bounded scores, no logit explosion."""
    q_n  = F.normalize(q,  dim=-1)   # (N, D)
    kv_n = F.normalize(kv, dim=-1)   # (M, D)
    scores = torch.einsum('vd,md->vm', q_n, kv_n)   # (N, M) in [-1, 1]
    attn   = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(
        torch.einsum('vm,md->vd', attn, kv), nan=0.0)   # (N, D)


def slat_avg_v1(data, t, k):
    """
    V1 — Naive average (PI spec):
      K̂ = V̂ = mean(z_{t-k}, ..., z_{t+k})     average K and V across window
      z̃_t = softmax(Q_t @ K̂ᵀ / √D) @ V̂        spatial attention with avg K,V
    Q is from the current frame; K,V are the time-averaged features.
    """
    T, N, D = data.shape
    lo, hi  = max(0, t - k), min(T - 1, t + k)
    K_avg   = data[lo:hi + 1].mean(dim=0)    # (N, D) — averaged K and V
    return _cosine_attn(data[t], K_avg)       # spatial attention: Q_t vs avg K,V


# ─────────────────────────────────────────────────────────────────────────────
# V2: Temporal THEN Spatial
# ─────────────────────────────────────────────────────────────────────────────

def slat_temporal_then_spatial_v2(data, t, k, alpha=1.0):
    """
    V2 — Temporal → Spatial:
      Stage 1 (temporal, per-voxel):
        For each voxel v, attend over its W temporal versions:
          scores_t[v, t'] = z_t[v] · z_{t'}[v] / √D    (N, W)
          α boost on current frame before softmax
          K̂_v = V̂_v = Σ_t' softmax_t[v,t'] · z_{t'}[v]
      Stage 2 (spatial):
        z̃_t = softmax(z_t @ K̂ᵀ / √D) @ V̂
    """
    T, N, D = data.shape
    lo, hi  = max(0, t - k), min(T - 1, t + k)
    window  = data[lo:hi + 1]                        # (W, N, D)
    q       = data[t]                                 # (N, D)

    # Stage 1: per-voxel temporal attention (cosine similarity)
    q_n      = F.normalize(q,      dim=-1)         # (N, D)
    win_n    = F.normalize(window, dim=-1)          # (W, N, D)
    scores_t = torch.einsum('nd,wnd->nw', q_n, win_n)   # (N, W) in [-1, 1]
    if alpha != 1.0:
        scores_t[:, t - lo] = scores_t[:, t - lo] * alpha
    attn_t   = torch.softmax(scores_t, dim=-1)            # (N, W)
    kv_blend = torch.nan_to_num(
        torch.einsum('nw,wnd->nd', attn_t, window), nan=0.0)   # (N, D)

    # Stage 2: spatial attention on temporally-blended K, V (cosine)
    return _cosine_attn(q, kv_blend)


# ─────────────────────────────────────────────────────────────────────────────
# V2b: Spatial THEN Temporal
# ─────────────────────────────────────────────────────────────────────────────

def slat_spatial_then_temporal_v2b(data, t, k, alpha=1.0):
    """
    V2b — Spatial → Temporal:
      Stage 1 (spatial, within frame t):
        spatial_out = softmax(z_t @ z_tᵀ / √D) @ z_t
      Stage 2 (temporal, per-voxel, using spatial_out as query):
        scores_t[v, t'] = spatial_out[v] · z_{t'}[v] / √D    (N, W)
        α boost on current frame before softmax
        z̃_t[v] = Σ_t' softmax_t[v,t'] · z_{t'}[v]
    """
    T, N, D = data.shape
    lo, hi  = max(0, t - k), min(T - 1, t + k)
    window  = data[lo:hi + 1]                          # (W, N, D)
    q       = data[t]                                   # (N, D)

    # Stage 1: spatial self-attention within frame t (cosine)
    spatial_out = _cosine_attn(q, q)   # (N, D)

    # Stage 2: per-voxel temporal attention using spatial_out as query (cosine)
    sp_n     = F.normalize(spatial_out, dim=-1)          # (N, D)
    win_n    = F.normalize(window,      dim=-1)          # (W, N, D)
    scores_t = torch.einsum('nd,wnd->nw', sp_n, win_n)  # (N, W) in [-1, 1]
    if alpha != 1.0:
        scores_t[:, t - lo] = scores_t[:, t - lo] * alpha
    attn_t   = torch.softmax(scores_t, dim=-1)           # (N, W)
    return torch.nan_to_num(
        torch.einsum('nw,wnd->nd', attn_t, window), nan=0.0)  # (N, D)


# ─────────────────────────────────────────────────────────────────────────────
# V3: Joint temporal+spatial in one softmax
# ─────────────────────────────────────────────────────────────────────────────

def slat_joint_v3(data, t, k, alpha=1.0):
    """
    V3 — Joint:
      Pool all window frames into one key/value bank: K_all (W·N, D)
      α boost on current frame's block before softmax
      z̃_t = softmax(z_t @ K_allᵀ / √D) @ K_all
      Single softmax over all frames AND all voxel positions simultaneously.
    """
    T, N, D = data.shape
    lo, hi  = max(0, t - k), min(T - 1, t + k)
    window  = data[lo:hi + 1]                          # (W, N, D)
    W       = window.shape[0]
    q       = data[t]                                   # (N, D)
    kv_all  = window.reshape(W * N, D)                 # (W*N, D)

    q_n     = F.normalize(q,      dim=-1)   # (N, D)
    kv_n    = F.normalize(kv_all, dim=-1)  # (W*N, D)
    scores  = torch.einsum('vd,md->vm', q_n, kv_n)   # (N, W*N) in [-1, 1]
    if alpha != 1.0:
        t_off = (t - lo) * N
        scores[:, t_off:t_off + N] = scores[:, t_off:t_off + N] * alpha
    attn = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(
        torch.einsum('vm,md->vd', attn, kv_all), nan=0.0)    # (N, D)


# ─────────────────────────────────────────────────────────────────────────────
# Runner: apply one mode to all frames, decode, render, evaluate
# ─────────────────────────────────────────────────────────────────────────────

def run_mode(pipeline, slats_np, coords_np, mode, k, alpha, blend,
             out_dir, gt_dir, trellis_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    data   = torch.from_numpy(slats_np).to(device)     # (T, N, D)
    coords = torch.from_numpy(coords_np).to(device)    # (N, 4)
    T      = data.shape[0]

    extr, intr = build_camera()
    rows = []

    for t in range(T):
        with torch.no_grad():
            if mode == 'avg':
                z_smooth = slat_avg_v1(data, t, k)
            elif mode == 'temporal_spatial':
                z_smooth = slat_temporal_then_spatial_v2(data, t, k, alpha)
            elif mode == 'spatial_temporal':
                z_smooth = slat_spatial_then_temporal_v2b(data, t, k, alpha)
            elif mode == 'joint':
                z_smooth = slat_joint_v3(data, t, k, alpha)
            else:
                raise ValueError(f'Unknown mode: {mode}')

            # Blend: preserve current-frame appearance while adding smoothing.
            # blend=1.0 → pure smoothed; blend=0.0 → pure original (no change)
            z_final = (1.0 - blend) * data[t] + blend * z_smooth
            color_u8 = decode_and_render(pipeline, z_final, coords, extr, intr)

        gt      = load_gt(t + 1, device)
        color_f = torch.from_numpy(
            color_u8.astype(np.float32) / 255.).permute(2, 0, 1).to(device)
        img_psnr = psnr(color_f, gt)
        rows.append({'t': t + 1, 'psnr': round(img_psnr, 3)})

        Image.fromarray(color_u8).save(out_dir / f'frame_{t + 1:03d}.png')
        gt_path = gt_dir / f'frame_{t + 1:03d}.png'
        if not gt_path.exists():
            Image.fromarray(to_uint8(gt)).save(gt_path)

        if (t + 1) % 25 == 0 or t == 0:
            print(f'  t={t + 1:03d}/{T}  PSNR={img_psnr:.2f} dB')

    avg_psnr = np.mean([r['psnr'] for r in rows])
    tag = f'{mode}_k{k}'
    print(f'{tag}  avg PSNR: {avg_psnr:.2f} dB')
    save_csv(rows, out_dir / 'psnr.csv')
    make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
    make_comparison_video(gt_dir, trellis_dir, out_dir, out_dir / 'comparison.mp4')
    return avg_psnr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',     type=int,   default=1,
                        help='Temporal window half-size')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Current-frame key boost in temporal attention. '
                             '1.0 = no boost. Higher = less smoothing.')
    parser.add_argument('--blend', type=float, default=0.5,
                        help='How much of the smoothed features to use. '
                             '0.0=no smoothing, 1.0=fully smoothed (may lose color).')
    parser.add_argument('--mode',  type=str,   default='all',
                        choices=['avg', 'temporal_spatial', 'spatial_temporal',
                                 'joint', 'all'],
                        help='Smoothing mode')
    args = parser.parse_args()

    ALL_MODES = ['avg', 'temporal_spatial', 'spatial_temporal', 'joint']
    modes_to_run = ALL_MODES if args.mode == 'all' else [args.mode]

    print(f'Loading TRELLIS pipeline: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    for model in pipeline.models.values():
        for p in model.parameters():
            p.requires_grad_(False)
    print('Pipeline loaded.')

    slats_np, coords_np = load_slat_cache()

    # ── Sanity check: decode (mesh) & render original SLaT frame 0 ─────────────
    print('=== Sanity check: mesh decode + render SLaT frame 0 ===')
    _dev    = torch.device('cuda')
    _feats  = torch.from_numpy(slats_np[0]).to(_dev)
    _coords = torch.from_numpy(coords_np).to(_dev)
    _extr, _intr = build_camera()
    try:
        _color = decode_and_render(pipeline, _feats, _coords, _extr, _intr)
        print(f'  sanity render: OK  shape={_color.shape}  '
              f'min={_color.min()}  max={_color.max()}')
    except Exception as e:
        print(f'  sanity render: FAILED → {e}')
        raise

    del _feats, _coords
    torch.cuda.empty_cache()
    # ───────────────────────────────────────────────────────────────────────────

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_dir      = OUT_DIR / 'gt'
    trellis_dir = OUT_DIR / 'trellis_renders'
    gt_dir.mkdir(parents=True, exist_ok=True)
    prepare_trellis_renders(trellis_dir)

    results = {}
    for mode in modes_to_run:
        tag     = f'{mode}_k{args.k}_b{args.blend}'
        out_dir = OUT_DIR / tag
        print(f'\n=== Phase 8 | {tag}  alpha={args.alpha}  blend={args.blend} ===')
        sys.stdout.flush()
        avg_psnr        = run_mode(
            pipeline, slats_np, coords_np,
            mode, args.k, args.alpha, args.blend,
            out_dir, gt_dir, trellis_dir,
        )
        results[tag] = avg_psnr

    print('\n=== Summary ===')
    for name, v in results.items():
        print(f'  {name:30s}: {v:.2f} dB')
    print(f'\nOutputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
