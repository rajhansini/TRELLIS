"""
Phase 7 — Temporal Attention Smoothing of SLAT Features
=========================================================
Motivation: Phase 5 per-frame optimization produces textures that match the
video frame-by-frame but flicker because each frame is optimized independently.
This phase adds cross-frame information via temporal attention over the SLAT
latent features.

The SLAT latent z_t has shape (N_voxels, feat_dim=8) with fixed voxel
coordinates across all frames.  For each frame t we build a temporal window
[t-k, ..., t, ..., t+k] and apply attention:

  Three smoothing modes (--mode):

  avg:
    z_hat_t = mean(z_{t-k}, ..., z_{t+k})
    No parameters. Baseline.

  attn_id:
    Identity-projection temporal attention (no training).
    Q_t(v)  = z_t[v]             (N_voxels, feat_dim)
    K_{t'}(v) = z_{t'}[v]
    V_{t'}(v) = z_{t'}[v]
    scores[v, t'] = dot(Q_t[v], K_{t'}[v]) / sqrt(feat_dim)
    attn = softmax(scores, dim=-1)   over t' window
    z_hat_t[v] = sum_{t'} attn[v, t'] * V_{t'}[v]

  attn_learned:
    Same structure but W_Q, W_K, W_V (feat_dim × feat_dim) are learned by
    minimising render loss against GT video frames.  Only 3 * 8*8 = 192
    parameters.  Optionally also fine-tune z_t (--ft_slat).

  You can run all three in sequence with --mode all.

Input:
  data/dynamic_sequences/trellis_seq/frame_0001/latent.npz   (for fixed coords)
  data/dynamic_sequences/phase0_slats.npz (cached after first run)
  trellis_150_frames/frame_XXXX/renders/front.png            (GT video frames)

Output: experiments/results/phase7/
  slat_cache.npz          — Phase 0 SLAT feats, shape (150, N_vox, 8)
  frames/<mode>_NNN.png   — rendered frame
  frames/gt_NNN.png       — GT frame
  comparison_<mode>.mp4   — GT | Phase0 | <mode>
  psnr_<mode>.csv         — per-frame PSNR

Run:
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase7/phase7.py [--k 1] [--mode all] [--n_learn 200]
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils
from trellis.renderers import GaussianRenderer

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
PHASE0_DIR     = REPO_ROOT / 'experiments' / 'results' / 'phase0'
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase7'
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
SLAT_SEQ_DIR   = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
NOISE_SEED = 42
RENDER_RES = 512
FLOW_STEPS = 25
FEAT_DIM   = 8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fixed_coords(device):
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


def load_gt(frame_idx, device):
    path = FRAMES_150_DIR / f'frame_{frame_idx:04d}' / 'renders' / 'front.png'
    img  = Image.open(path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.).permute(2, 0, 1).to(device)


def build_camera():
    """Returns (extrinsics_list, intrinsics_list) for the front view.
    yaw=0 (facing +Y axis), pitch=0.25 matches TRELLIS render_video frame-0 convention.
    """
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [0.25], 2.0, 40.0,
    )


def render_feats(pipeline, feats, coords):
    """Decode SLAT feats → Gaussians → render front view via render_utils. Returns (H,W,3) uint8."""
    slat = sp.SparseTensor(feats=feats, coords=coords)
    with torch.no_grad():
        decoded  = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]
    extr, intr = build_camera()
    frames = render_utils.render_frames(
        gaussian, extr, intr,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return frames['color'][0]  # (H,W,3) uint8


def render_feats_grad(pipeline, renderer, feats, coords, extr, intr):
    """Differentiable render — keeps gradient graph. Returns (3,H,W) float32."""
    slat    = sp.SparseTensor(feats=feats, coords=coords)
    decoded = pipeline.decode_slat(slat, ['gaussian'])
    return renderer.render(decoded['gaussian'][0], extr, intr)['color']


def to_uint8(tensor):
    return (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def psnr(pred, gt):
    mse = float(F.mse_loss(pred.detach(), gt.detach()).item())
    return 10 * math.log10(1.0 / (mse + 1e-10))


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Cache decoded Gaussian parameters for all 150 frames
# ─────────────────────────────────────────────────────────────────────────────

GAUSS_KEYS = ['_xyz', '_scaling', '_rotation', '_features_dc', '_opacity']

def gaussian_to_np(g):
    """Extract raw (pre-activation) Gaussian params as dict of numpy arrays."""
    return {k: getattr(g, k).detach().cpu().float().numpy() for k in GAUSS_KEYS}

def np_to_gaussian(params_np, rep_config, device):
    """Reconstruct a Gaussian object from cached numpy params."""
    from trellis.representations import Gaussian
    g = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=rep_config['3d_filter_kernel_size'],
        scaling_bias=rep_config['scaling_bias'],
        opacity_bias=rep_config['opacity_bias'],
        scaling_activation=rep_config['scaling_activation'],
    )
    for k in GAUSS_KEYS:
        t = torch.from_numpy(params_np[k].copy()).to(device)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if k == '_scaling':
            t = t.clamp(-10.0, 4.0)
        setattr(g, k, t)
    return g

def compute_phase0_gaussians(pipeline, device, cache_path):
    """
    Run Phase 0 for all 150 frames, decode each SLAT to Gaussians, and cache
    the raw Gaussian parameters (xyz, scaling, rotation, features_dc, opacity).
    Returns a dict mapping key → (T, N_gauss, param_dim) float32 numpy arrays.
    """
    if cache_path.exists():
        print(f'Loading cached Gaussian params from {cache_path}')
        data = np.load(cache_path)
        arrays = {k: data[k] for k in GAUSS_KEYS}
        # Validate cache — report any non-finite values and their range.
        for k, arr in arrays.items():
            bad = ~np.isfinite(arr)
            if bad.any():
                print(f'  WARNING: {bad.sum()} non-finite values in cached {k} '
                      f'(out of {arr.size})')
            print(f'  {k}: min={arr[np.isfinite(arr)].min():.4f}  '
                  f'max={arr[np.isfinite(arr)].max():.4f}  '
                  f'nan={np.isnan(arr).sum()}  inf={np.isinf(arr).sum()}')
        return arrays

    tokens_path = PHASE0_DIR / 'tokens.npz'
    if not tokens_path.exists():
        raise FileNotFoundError(f'Run Phase 0 first: {tokens_path}')
    data_tok = np.load(tokens_path)
    T_0   = torch.from_numpy(data_tok['T_0'].astype(np.float32)).to(device)
    T_150 = torch.from_numpy(data_tok['T_150'].astype(np.float32)).to(device)

    coords = fixed_coords(device)

    # First pass: get shapes
    alpha  = 0.0
    T_t    = T_0.clone()
    cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}
    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        slat = pipeline.sample_slat(cond_t, coords, sampler_params={'steps': FLOW_STEPS})
        g0   = pipeline.decode_slat(slat, ['gaussian'])['gaussian'][0]
    shapes = {k: getattr(g0, k).shape for k in GAUSS_KEYS}
    print(f'Gaussian shapes: { {k: v for k, v in shapes.items()} }')

    arrays = {k: np.zeros((N_FRAMES, *shapes[k]), dtype=np.float32) for k in GAUSS_KEYS}
    arrays_np = gaussian_to_np(g0)
    for k in GAUSS_KEYS:
        arrays[k][0] = arrays_np[k]

    print(f'Computing Phase 0 Gaussians for {N_FRAMES} frames ...')
    for t in range(2, N_FRAMES + 1):
        alpha  = (t - 1) / (N_FRAMES - 1)
        T_t    = (1.0 - alpha) * T_0 + alpha * T_150
        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

        torch.manual_seed(NOISE_SEED)
        with torch.no_grad():
            slat = pipeline.sample_slat(cond_t, coords, sampler_params={'steps': FLOW_STEPS})
            g    = pipeline.decode_slat(slat, ['gaussian'])['gaussian'][0]

        gp = gaussian_to_np(g)
        for k in GAUSS_KEYS:
            arrays[k][t - 1] = gp[k]

        if t % 25 == 0:
            print(f'  t={t:03d}/{N_FRAMES}  α={alpha:.3f}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    print(f'Cached: {cache_path}')
    return arrays


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Temporal smoothing on Gaussian parameter space
# ─────────────────────────────────────────────────────────────────────────────

def _concat_params(gauss_arrays):
    """Stack all Gaussian params into (T, N_gauss, D_total) for attention."""
    parts = []
    for k in GAUSS_KEYS:
        a = gauss_arrays[k]                        # (T, N, ...)
        T, N = a.shape[0], a.shape[1]
        parts.append(a.reshape(T, N, -1))
    return np.concatenate(parts, axis=-1)          # (T, N, D_total)

def _split_params(concat, gauss_arrays):
    """Inverse of _concat_params — returns dict matching gauss_arrays structure."""
    T, N, _ = concat.shape
    out, offset = {}, 0
    for k in GAUSS_KEYS:
        d = gauss_arrays[k][0].reshape(gauss_arrays[k].shape[1], -1).shape[-1]
        chunk = concat[:, :, offset:offset + d]
        out[k] = chunk.reshape(gauss_arrays[k].shape)
        offset += d
    return out

def temporal_avg(gauss_arrays, k):
    """Simple temporal moving average on all Gaussian params."""
    out = {}
    for key, arr in gauss_arrays.items():    # arr: (T, N, ...)
        T = arr.shape[0]
        smoothed = np.zeros_like(arr)
        for t in range(T):
            lo, hi = max(0, t - k), min(T - 1, t + k)
            smoothed[t] = arr[lo:hi + 1].mean(axis=0)
        out[key] = smoothed
    return out

def temporal_attn_identity(gauss_arrays, k):
    """
    Identity-QKV temporal attention on concatenated Gaussian params.
    (T, N_gauss, D_total) → same shape, smoothed over time window.
    """
    concat  = _concat_params(gauss_arrays)   # (T, N, D)
    T, N, D = concat.shape
    data_t  = torch.from_numpy(concat)
    out     = torch.zeros_like(data_t)
    scale   = D ** -0.5

    for t in range(T):
        lo, hi  = max(0, t - k), min(T - 1, t + k)
        window  = data_t[lo:hi + 1]                              # (W, N, D)
        scores  = torch.einsum('nd,wnd->nw', data_t[t], window) * scale  # (N, W)
        attn    = torch.softmax(scores, dim=-1)
        out[t]  = torch.einsum('nw,wnd->nd', attn, window)

    return _split_params(out.numpy().astype(np.float32), gauss_arrays)


def temporal_spatial_joint(gauss_arrays, k, alpha=2.0, n_per_voxel=32):
    """
    Version 3: Joint temporal+spatial attention (one-stage), at voxel level.

    Geometry (_xyz, _scaling, _rotation): temporally averaged over the k-window.
    Appearance (_features_dc, _opacity): joint voxel-level temporal+spatial attention.

    Using avg geometry is necessary because the raw phase-0 cache contains geometric
    states that cause int32 overflow in the CUDA rasterizer's tile-pair counter.
    avg_k5 avoids this; we adopt the same geometry stabilization here so that the
    unique contribution of this mode — the joint voxel-level appearance attention —
    can be evaluated cleanly.
    """
    dev     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    concat  = _concat_params(gauss_arrays)   # (T, N, D=14)
    T, N, D = concat.shape
    V       = N // n_per_voxel               # ~6281 voxels
    data_t  = torch.from_numpy(concat).to(dev)

    # Compute appearance dim offsets: _features_dc + _opacity = last 4 dims
    a_start, offset = None, 0
    for key in GAUSS_KEYS:
        d = gauss_arrays[key][0].reshape(gauss_arrays[key].shape[1], -1).shape[1]
        if key == '_features_dc':
            a_start = offset
        offset += d
    a_end = D  # _opacity is the last key so appearance runs a_start:D

    # Mean-pool to voxel level: (T, V, D)
    vox = data_t.view(T, V, n_per_voxel, D).mean(dim=2)

    scale   = D ** -0.5
    out_vox = torch.zeros_like(vox)
    with torch.no_grad():
        for t in range(T):
            lo, hi   = max(0, t - k), min(T - 1, t + k)
            window   = vox[lo:hi + 1]             # (W, V, D)
            W        = window.shape[0]
            q        = vox[t]                     # (V, D)
            kv       = window.reshape(W * V, D)   # (W*V, D)
            # alpha boost on current frame K only — prevents over-smoothing
            t_offset = (t - lo) * V
            k_boost  = kv.clone()
            k_boost[t_offset:t_offset + V] *= alpha
            scores   = torch.einsum('vd,md->vm', q, k_boost) * scale  # (V, W*V)
            attn     = torch.softmax(scores, dim=-1)                   # joint softmax
            out_vox[t] = torch.einsum('vm,md->vd', attn, kv)

    # Expand voxel attention output to per-Gaussian level
    appear_delta = (out_vox[..., a_start:a_end] - vox[..., a_start:a_end])  # (T, V, A)
    appear_delta = appear_delta.unsqueeze(2).expand(T, V, n_per_voxel, a_end - a_start)
    appear_delta = appear_delta.reshape(T, N, a_end - a_start)

    out = data_t.clone()
    out[:, :, a_start:a_end] += appear_delta   # joint attention appearance

    out_np = out.cpu().numpy().astype(np.float32)
    smoothed = _split_params(out_np, gauss_arrays)

    # Replace geometry dims with temporally averaged values — this keeps the
    # geometry rasterizer-safe (same stabilization as avg_k5 which works).
    geom_keys = ['_xyz', '_scaling', '_rotation']
    for key in geom_keys:
        arr = gauss_arrays[key]          # (T, N, ...)
        avg = np.zeros_like(arr)
        for t in range(T):
            lo, hi = max(0, t - k), min(T - 1, t + k)
            avg[t] = arr[lo:hi + 1].mean(axis=0)
        smoothed[key] = avg

    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Learned temporal attention module (operates on Gaussian params)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionModule(nn.Module):
    """
    Learnable temporal attention over concatenated Gaussian parameters.
    feat_dim = D_total (xyz+scaling+rotation+color+opacity per Gaussian).
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.W_Q  = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_K  = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_V  = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_out = nn.Linear(feat_dim, feat_dim, bias=False)
        self.scale = feat_dim ** -0.5
        nn.init.eye_(self.W_Q.weight)
        nn.init.eye_(self.W_K.weight)
        nn.init.eye_(self.W_V.weight)
        nn.init.zeros_(self.W_out.weight)  # zero init → residual starts as identity

    def forward(self, data: torch.Tensor, t: int, k: int) -> torch.Tensor:
        """data: (T, N_gauss, D).  Returns smoothed frame t: (N_gauss, D)."""
        T  = data.shape[0]
        lo, hi = max(0, t - k), min(T - 1, t + k)
        window = data[lo:hi + 1]
        q   = self.W_Q(data[t])
        ks  = self.W_K(window)
        vs  = self.W_V(window)
        scores = torch.einsum('nd,wnd->nw', q, ks) * self.scale
        attn   = torch.softmax(scores, dim=-1)
        agg    = torch.einsum('nw,wnd->nd', attn, vs)
        return self.W_out(agg) + data[t]


def _scaling_slice(gauss_arrays):
    """Return (start, end) slice indices for _scaling within the concat vector."""
    offset = 0
    for key in GAUSS_KEYS:
        d = gauss_arrays[key][0].reshape(gauss_arrays[key].shape[1], -1).shape[1]
        if key == '_scaling':
            return offset, offset + d
        offset += d
    raise RuntimeError('_scaling not found in GAUSS_KEYS')


def train_temporal_attn(
    gauss_arrays, rep_config, device,
    k, n_steps=200, lr=1e-3, log_every=20,
):
    """
    Train temporal attention weights directly in Gaussian parameter space.
    Differentiable path: smoothed params → reconstruct Gaussian → render → loss vs GT.
    Returns smoothed gauss_arrays dict.
    """
    concat = _concat_params(gauss_arrays)         # (T, N, D)
    T, N, D = concat.shape
    data_t  = torch.from_numpy(concat).to(device) # (T, N, D)

    sc_lo, sc_hi = _scaling_slice(gauss_arrays)   # indices of _scaling in concat

    module    = TemporalAttentionModule(feat_dim=D).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    extr, intr = build_camera()
    extr_single = extr[0]
    intr_single = intr[0]

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = RENDER_RES
    renderer.rendering_options.bg_color = (1.0, 1.0, 1.0)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6

    print(f'Training temporal attention (Gauss space): k={k}, n_steps={n_steps}, lr={lr}')
    for step in range(n_steps):
        optimizer.zero_grad()

        batch_size = min(4, T)
        frame_losses = []
        for t in torch.randint(0, T, (batch_size,)).tolist():
            smoothed_t = module(data_t, t, k)          # (N, D)
            # Clamp _scaling to prevent rasterizer overflow
            smoothed_t = torch.cat([
                smoothed_t[:, :sc_lo],
                smoothed_t[:, sc_lo:sc_hi].clamp(-10.0, 4.0),
                smoothed_t[:, sc_hi:],
            ], dim=1)
            params_t = _split_concat_to_params(smoothed_t, gauss_arrays, device)
            g = np_to_gaussian_from_tensors(params_t, rep_config, device)
            color = renderer.render(g, extr_single, intr_single)['color']
            gt    = load_gt(t + 1, device)
            frame_losses.append(F.mse_loss(color, gt))

        loss = sum(frame_losses) / len(frame_losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % log_every == 0 or step == 0:
            print(f'  step={step+1:04d}/{n_steps}  loss={loss.item():.6f}')

    print('Applying learned temporal attention to all frames ...')
    out = torch.zeros_like(data_t)
    with torch.no_grad():
        for t in range(T):
            smoothed_t = module(data_t, t, k)
            smoothed_t = torch.cat([
                smoothed_t[:, :sc_lo],
                smoothed_t[:, sc_lo:sc_hi].clamp(-10.0, 4.0),
                smoothed_t[:, sc_hi:],
            ], dim=1)
            out[t] = smoothed_t
    return _split_params(out.detach().cpu().numpy().astype(np.float32), gauss_arrays)


def _split_concat_to_params(concat_t, gauss_arrays, device):
    """Split a single-frame (N, D) tensor back into per-key tensors."""
    out, offset = {}, 0
    for k in GAUSS_KEYS:
        shape = gauss_arrays[k].shape[1:]    # (N, ...) → (...)
        N = gauss_arrays[k].shape[1]
        d = int(np.prod(shape[1:]) if len(shape) > 1 else 1) if len(shape) > 0 else 1
        # correct dim: flatten shape except first dim
        orig_flat = gauss_arrays[k][0].reshape(N, -1).shape[1]
        chunk = concat_t[:, offset:offset + orig_flat]
        out[k] = chunk.reshape(N, *gauss_arrays[k].shape[2:])
        offset += orig_flat
    return out


def np_to_gaussian_from_tensors(params_tensors, rep_config, device):
    """Reconstruct Gaussian from dict of tensors (differentiable)."""
    from trellis.representations import Gaussian
    g = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=rep_config['3d_filter_kernel_size'],
        scaling_bias=rep_config['scaling_bias'],
        opacity_bias=rep_config['opacity_bias'],
        scaling_activation=rep_config['scaling_activation'],
    )
    for k, v in params_tensors.items():
        setattr(g, k, v)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Render and compare
# ─────────────────────────────────────────────────────────────────────────────

def render_and_save(gauss_arrays, rep_config, device, out_dir, gt_dir,
                    start=1, end=N_FRAMES):
    """
    Render all frames into out_dir/frame_NNN.png.
    GT frames saved to gt_dir/frame_NNN.png (skipped if already exists).
    Returns per-frame PSNR list.
    """
    torch.cuda.empty_cache()
    extr, intr = build_camera()
    rows = []
    for t in range(start, end + 1):
        i = t - 1
        params_np = {k: gauss_arrays[k][i] for k in GAUSS_KEYS}
        g        = np_to_gaussian(params_np, rep_config, device)
        frames   = render_utils.render_frames(
            g, extr, intr,
            options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
            verbose=False,
        )
        color_u8 = frames['color'][0]  # (H, W, 3) uint8
        gt       = load_gt(t, device)

        color_f  = torch.from_numpy(color_u8.astype(np.float32) / 255.).permute(2, 0, 1).to(device)
        img_psnr = psnr(color_f, gt)
        rows.append({'t': t, 'psnr': round(img_psnr, 3)})

        Image.fromarray(color_u8).save(out_dir / f'frame_{t:03d}.png')
        gt_path = gt_dir / f'frame_{t:03d}.png'
        if not gt_path.exists():
            Image.fromarray(to_uint8(gt)).save(gt_path)

        if t % 25 == 0 or t == start:
            print(f'  t={t:03d}  PSNR={img_psnr:.2f} dB')
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _rep_config_from_disk():
    """Load rep_config from the cached pipeline JSON without loading the model."""
    import json
    json_path = (Path(os.environ.get('HF_HOME',
                  Path.home() / '.cache' / 'huggingface'))
                 / 'hub' / 'models--microsoft--TRELLIS-image-large'
                 / 'snapshots')
    # find the single snapshot directory
    snaps = list(json_path.glob('*')) if json_path.exists() else []
    if not snaps:
        # also try /net/scratch default
        alt = Path('/net/scratch/rajhansini/.cache/huggingface/hub'
                   '/models--microsoft--TRELLIS-image-large/snapshots')
        snaps = list(alt.glob('*')) if alt.exists() else []
    if not snaps:
        return None
    cfg_path = snaps[0] / 'ckpts' / 'slat_dec_gs_swin8_B_64l8gs32_fp16.json'
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        d = json.load(f)
    return d['args']['representation_config']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',        type=int,   default=1,
                        help='Temporal window half-size (use k=1 for window=3)')
    parser.add_argument('--mode',     type=str,   default='all',
                        choices=['avg', 'attn_id', 'joint', 'attn_learned', 'all'],
                        help='Smoothing mode(s) to run')
    parser.add_argument('--n_learn',  type=int,   default=200,
                        help='Training steps for attn_learned')
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--ft_slat',  action='store_true',
                        help='Also fine-tune z_t features during training')
    parser.add_argument('--no_phase0', action='store_true',
                        help='Skip Phase 0 rendering (only do smoothed modes)')
    args = parser.parse_args()

    cache_path = OUT_DIR / 'gaussian_cache.npz'
    modes_to_run = ['avg', 'attn_id', 'joint', 'attn_learned'] if args.mode == 'all' else [args.mode]
    needs_pipeline = not cache_path.exists()

    render_device = torch.device('cuda')
    pipeline = None

    if needs_pipeline:
        # ── Load pipeline ─────────────────────────────────────────────────────
        print(f'Loading TRELLIS: {PRETRAINED}')
        sys.stdout.flush()
        pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
        pipeline.cuda()
        device = pipeline.device
        for model in pipeline.models.values():
            for p in model.parameters():
                p.requires_grad_(False)
        rep_config = pipeline.models['slat_decoder_gs'].rep_config
    else:
        # Cache already exists and we only need avg/attn_id — read rep_config
        # from the model JSON on disk, no GPU needed for model loading.
        print('Gaussian cache already exists — skipping pipeline load.')
        sys.stdout.flush()
        rep_config = _rep_config_from_disk()
        if rep_config is None:
            raise RuntimeError('Cannot read rep_config from disk and pipeline '
                               'load was skipped. Delete gaussian_cache.npz to '
                               'force a full pipeline run.')
        device = render_device

    print(f'k={args.k}   mode={args.mode}')
    sys.stdout.flush()

    # ── Cache decoded Gaussian params for all frames ──────────────────────────
    gauss_arrays = compute_phase0_gaussians(pipeline, device, cache_path)
    sample_shape = gauss_arrays['_xyz'].shape
    print(f'Gaussian cache: T={sample_shape[0]}  N_gauss={sample_shape[1]}')
    sys.stdout.flush()

    # ── Free pipeline GPU memory before rendering ─────────────────────────────
    if pipeline is not None and 'attn_learned' not in modes_to_run:
        print('Moving pipeline to CPU to free GPU VRAM for rendering...')
        for model in pipeline.models.values():
            model.cpu()
        pipeline = None
        torch.cuda.empty_cache()
    elif pipeline is not None:
        print('Moving pipeline to CPU to free GPU VRAM for rendering...')
        for model in pipeline.models.values():
            model.cpu()
        torch.cuda.empty_cache()

    # Shared GT dir (written once, reused across runs)
    gt_dir = OUT_DIR / 'gt'
    gt_dir.mkdir(parents=True, exist_ok=True)

    # ── Render Phase 0 baseline ───────────────────────────────────────────────
    if not args.no_phase0:
        p0_dir = OUT_DIR / 'phase0'
        p0_dir.mkdir(parents=True, exist_ok=True)
        print('\n=== Rendering Phase 0 baseline ===')
        p0_rows = render_and_save(
            gauss_arrays, rep_config, render_device,
            p0_dir, gt_dir,
        )
        avg_psnr_p0 = np.mean([r['psnr'] for r in p0_rows])
        print(f'Phase 0 avg PSNR: {avg_psnr_p0:.2f} dB')
        _save_csv(p0_rows, p0_dir / 'psnr.csv')
        make_video(p0_dir, 'frame_%03d.png', p0_dir / 'render.mp4')

    all_results = {}

    # ── Mode: avg (Version 1) ─────────────────────────────────────────────────
    if 'avg' in modes_to_run:
        tag = f'avg_k{args.k}'
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Version 1: avg  k={args.k} ===')
        smoothed = temporal_avg(gauss_arrays, args.k)
        rows = render_and_save(smoothed, rep_config, render_device, out_dir, gt_dir)
        avg_psnr = np.mean([r['psnr'] for r in rows])
        print(f'avg k={args.k}  avg PSNR: {avg_psnr:.2f} dB')
        _save_csv(rows, out_dir / 'psnr.csv')
        all_results[tag] = avg_psnr
        make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
        _make_comparison_video(gt_dir, OUT_DIR / 'phase0', out_dir, tag)

    # ── Mode: attn_id (Version 2) ─────────────────────────────────────────────
    if 'attn_id' in modes_to_run:
        tag = f'attn_id_k{args.k}'
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Version 2: attn_id  k={args.k} ===')
        smoothed = temporal_attn_identity(gauss_arrays, args.k)
        rows = render_and_save(smoothed, rep_config, render_device, out_dir, gt_dir)
        avg_psnr = np.mean([r['psnr'] for r in rows])
        print(f'attn_id k={args.k}  avg PSNR: {avg_psnr:.2f} dB')
        _save_csv(rows, out_dir / 'psnr.csv')
        all_results[tag] = avg_psnr
        make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
        _make_comparison_video(gt_dir, OUT_DIR / 'phase0', out_dir, tag)

    # ── Mode: joint (Version 3) ───────────────────────────────────────────────
    if 'joint' in modes_to_run:
        tag = f'joint_k{args.k}'
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Version 3: joint temporal+spatial  k={args.k} ===')
        smoothed = temporal_spatial_joint(gauss_arrays, args.k)
        rows = render_and_save(smoothed, rep_config, render_device, out_dir, gt_dir)
        avg_psnr = np.mean([r['psnr'] for r in rows])
        print(f'joint k={args.k}  avg PSNR: {avg_psnr:.2f} dB')
        _save_csv(rows, out_dir / 'psnr.csv')
        all_results[tag] = avg_psnr
        make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
        _make_comparison_video(gt_dir, OUT_DIR / 'phase0', out_dir, tag)

    # ── Mode: attn_learned (Version 4 — learned) ─────────────────────────────
    if 'attn_learned' in modes_to_run:
        tag = f'attn_learned_k{args.k}'
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Version 4: attn_learned  k={args.k} ===')
        smoothed = train_temporal_attn(
            gauss_arrays, rep_config, render_device,
            k=args.k, n_steps=args.n_learn, lr=args.lr,
        )
        torch.cuda.empty_cache()
        rows = render_and_save(smoothed, rep_config, render_device, out_dir, gt_dir)
        avg_psnr = np.mean([r['psnr'] for r in rows])
        print(f'{tag}  avg PSNR: {avg_psnr:.2f} dB')
        _save_csv(rows, out_dir / 'psnr.csv')
        all_results[tag] = avg_psnr
        make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
        _make_comparison_video(gt_dir, OUT_DIR / 'phase0', out_dir, tag)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n=== Summary (avg PSNR across 150 frames) ===')
    if not args.no_phase0:
        print(f'  phase0 baseline:   {avg_psnr_p0:.2f} dB')
    for name, v in all_results.items():
        delta = f' (Δ={v - avg_psnr_p0:+.2f})' if not args.no_phase0 else ''
        print(f'  {name:30s}: {v:.2f} dB{delta}')
    print(f'\nOutputs: {OUT_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved: {path}')


def _make_comparison_video(gt_dir, phase0_dir, smoothed_dir, tag):
    """3-panel video: GT | Phase0 (flicker baseline) | smoothed."""
    try:
        out_path = smoothed_dir / 'comparison.mp4'
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '15',
            '-i', str(gt_dir       / 'frame_%03d.png'),
            '-i', str(phase0_dir   / 'frame_%03d.png'),
            '-i', str(smoothed_dir / 'frame_%03d.png'),
            '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
        ], check=True, capture_output=True)
        print(f'Saved: {out_path}')
    except Exception as e:
        print(f'Video failed ({tag}): {e}')


if __name__ == '__main__':
    main()
