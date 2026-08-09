"""
Phase 8 (GL) — Cross-Frame Attention on DINO Conditioning Tokens Inside G_L
=============================================================================
Fix: let frame t's G_L cross-attention also attend to DINO conditioning tokens
from neighboring frames t-k...t+k (instead of only its own DINO tokens).

DINO conditioning tokens are fixed during denoising, so blended contexts are
precomputed once per run (before the denoising loop) and passed to the standard
flow_model forward. The blend is: ctx = (1-blend)*DINO_t + blend*blended_ctx.

4 versions — all operating on DINO tokens (1, 1374, 1024) per frame:

V1 (avg):
    K̂ = V̂ = mean(DINO_{t-k}, ..., DINO_{t+k})   [simple average across window]
    G_L cross_attn: voxel Q vs K̂/V̂

V2 (temporal_spatial):
    Stage 1 — per DINO-token-position temporal blend:
        scores[n, w] = DINO_t[n] · DINO_w[n] / √D      (N, W)
        α-boost current frame in log space: scores[:, cur] += log(α)
        K̂[n] = Σ_w softmax(scores[n]) · DINO_w[n]       (N, D)
    Stage 2 — G_L cross_attn uses voxel Q vs K̂/V̂  [happens inside the block]

V2b (spatial_temporal):
    Stage 1 — spatial self-attention of DINO_t tokens:
        Q_ref = softmax(DINO_t @ DINO_tᵀ / √D) @ DINO_t
    Stage 2 — per DINO-token-position temporal blend using Q_ref:
        K̂[n] = Σ_w softmax(Q_ref[n] · DINO_w[n] / √D + log(α)) · DINO_w[n]
    [G_L cross_attn uses voxel Q vs K̂/V̂ — no third stage]

V3 (joint):
    ctx = cat(DINO_{t-k}, ..., DINO_{t+k})    → (1, W×1374, D)
    G_L cross_attn: voxel Q attends over all W×1374 DINO tokens jointly

blend: ctx_final = (1-blend)*DINO_t + blend*ctx   (applied per-token position)
       V3 doesn't support blend interpolation (different shape); uses full ctx.

α=1 (default) → log(1)=0 → no boost.

Run:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python \\
      experiments/phase8/phase8_gl.py --mode avg --k 1 --blend 0.5
"""

import os, sys, math, csv, subprocess, argparse
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase8_gl'
GT_VIDEO_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                      '/outputs/teapot_lava_kling_premium'
                      '/teapot_lava_kling_premium_front/all_frames_150')
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
RENDER_RES = 512
FLOW_STEPS = 25
NOISE_SEED = 42


# ── Attention helper ──────────────────────────────────────────────────────────

def _sdp_attn(q, kv):
    """Standard scaled dot-product attention.  Q: (N,D)  KV: (M,D) → (N,D)."""
    D      = q.shape[-1]
    scores = torch.einsum('nd,md->nm', q.float(), kv.float()) / math.sqrt(D)
    attn   = torch.softmax(scores, dim=-1)
    out    = torch.einsum('nm,md->nd', attn, kv.float())
    return torch.nan_to_num(out, nan=0.0).to(q.dtype)


# ── Blended DINO context per version ─────────────────────────────────────────
# all_conds : list of T tensors, each (1, 1374, D)
# Returns   : (1, N_ctx, D)  — N_ctx=1374 for V1/V2/V2b, W×1374 for V3

def blend_avg(all_conds, t_idx, k, alpha, blend):
    """V1: average DINO tokens across window."""
    T = len(all_conds)
    lo, hi = max(0, t_idx - k), min(T - 1, t_idx + k)
    window = torch.stack([all_conds[i][0] for i in range(lo, hi + 1)])  # (W, N, D)
    ctx    = window.mean(dim=0)                                           # (N, D)
    ctx    = (1 - blend) * all_conds[t_idx][0] + blend * ctx
    return ctx.unsqueeze(0)


def blend_temporal_spatial(all_conds, t_idx, k, alpha, blend):
    """V2: per-position temporal blend of DINO tokens using DINO_t as Q."""
    T = len(all_conds)
    lo, hi = max(0, t_idx - k), min(T - 1, t_idx + k)
    window = torch.stack([all_conds[i][0] for i in range(lo, hi + 1)])  # (W, N, D)
    W, N, D = window.shape
    cur    = t_idx - lo
    q      = all_conds[t_idx][0]                                          # (N, D)
    # Per DINO-token-position temporal attention
    scores = torch.einsum('nd,wnd->nw', q.float(), window.float()) / math.sqrt(D)
    scores[:, cur] = scores[:, cur] + math.log(alpha)   # log-space; α=1 → 0
    weights = torch.softmax(scores, dim=-1)              # (N, W)
    ctx     = torch.einsum('nw,wnd->nd', weights, window.float()).to(q.dtype)
    # Stage 2 (spatial cross-attn with voxel Q) happens inside G_L's cross_attn
    ctx = (1 - blend) * q + blend * ctx
    return ctx.unsqueeze(0)


def blend_spatial_temporal(all_conds, t_idx, k, alpha, blend):
    """V2b: spatial self-attention of DINO_t → temporal blend. No third stage."""
    T = len(all_conds)
    lo, hi = max(0, t_idx - k), min(T - 1, t_idx + k)
    window = torch.stack([all_conds[i][0] for i in range(lo, hi + 1)])  # (W, N, D)
    W, N, D = window.shape
    cur    = t_idx - lo
    q      = all_conds[t_idx][0]                                          # (N, D)
    # Stage 1: spatial self-attention of current frame's DINO tokens
    q_ref  = _sdp_attn(q, q)                                             # (N, D)
    # Stage 2: temporal blend using spatially-refined Q — no third stage
    scores  = torch.einsum('nd,wnd->nw', q_ref.float(), window.float()) / math.sqrt(D)
    scores[:, cur] = scores[:, cur] + math.log(alpha)
    weights = torch.softmax(scores, dim=-1)              # (N, W)
    ctx     = torch.einsum('nw,wnd->nd', weights, window.float()).to(q.dtype)
    ctx     = (1 - blend) * q + blend * ctx
    return ctx.unsqueeze(0)


def blend_joint(all_conds, t_idx, k, alpha, blend):
    """V3: concatenate all window DINO tokens → joint cross-attention in G_L."""
    T = len(all_conds)
    lo, hi  = max(0, t_idx - k), min(T - 1, t_idx + k)
    window  = torch.stack([all_conds[i][0] for i in range(lo, hi + 1)])  # (W, N, D)
    W, N, D = window.shape
    # blend not applicable (shape changes); full concatenated context used
    return window.reshape(1, W * N, D).to(all_conds[t_idx].dtype)


BLENDERS = {
    'avg':              blend_avg,
    'temporal_spatial': blend_temporal_spatial,
    'spatial_temporal': blend_spatial_temporal,
    'joint':            blend_joint,
}


# ── Denoising loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_all_frames_crossframe(pipeline, all_conds_np, coords, blender, k, alpha, blend,
                                  slat_feats_np=None,
                                  t_start=1.0,
                                  steps=FLOW_STEPS,
                                  cfg_strength=5.0,
                                  cfg_interval=(0.5, 1.0),
                                  rescale_t=3.0):
    """
    Denoise all T frames. Each frame's G_L cross-attention uses a blended DINO
    context computed from window frames instead of only its own DINO tokens.

    DINO tokens are fixed during denoising → blended contexts precomputed once.

    all_conds_np : list of T arrays (1, 1374, 1024)
    coords       : (N_vox, 4) int32 cuda tensor (batch col 0 prepended)
    Returns      : (T, N_vox, 8) float32 numpy — denormalized SLaT features
    """
    T          = len(all_conds_np)
    flow_model = pipeline.models['slat_flow_model']
    device     = next(flow_model.parameters()).device
    N_vox      = coords.shape[0]

    std  = torch.tensor(pipeline.slat_normalization['std'],  device=device)
    mean = torch.tensor(pipeline.slat_normalization['mean'], device=device)

    # ── Init: SDEdit from slat_cache or pure noise ────────────────────────────
    all_x = []
    for i in range(T):
        torch.manual_seed(NOISE_SEED + i)
        noise = torch.randn(N_vox, flow_model.in_channels, device=device)
        if slat_feats_np is not None and t_start < 1.0:
            x0_raw  = torch.from_numpy(slat_feats_np[i].astype(np.float32)).to(device)
            x0_norm = (x0_raw - mean) / std
            feats   = (1 - t_start) * x0_norm + t_start * noise
            if i == 0:
                print(f'[DEBUG] SDEdit t_start={t_start}  x0_norm mean={x0_norm.mean():.3f}  x_t mean={feats.mean():.3f}')
        else:
            feats = noise
        all_x.append(sp.SparseTensor(feats=feats, coords=coords))

    # ── DINO conditioning tensors ─────────────────────────────────────────────
    all_cond     = [torch.from_numpy(c.astype(np.float32)).to(device) for c in all_conds_np]
    all_neg_cond = [torch.zeros_like(c) for c in all_cond]

    c0 = all_cond[0]
    print(f'[DEBUG] raw DINO cond[0]:     shape={tuple(c0.shape)}  min={c0.min():.4f}  max={c0.max():.4f}  mean={c0.mean():.4f}  std={c0.std():.4f}  nan={c0.isnan().any()}')
    print(f'[DEBUG] raw DINO neg_cond[0]: all_zeros={all_neg_cond[0].abs().max().item() == 0}')
    print(f'[DEBUG] initial noise feats:  min={all_x[0].feats.min():.3f}  max={all_x[0].feats.max():.3f}  mean={all_x[0].feats.mean():.3f}')

    # ── Precompute blended contexts ONCE (DINO tokens fixed during denoising) ─
    print('Precomputing blended DINO contexts...')
    blended_cond     = [blender(all_cond,     i, k, alpha, blend) for i in range(T)]
    blended_neg_cond = [blender(all_neg_cond, i, k, alpha, blend) for i in range(T)]
    bc0 = blended_cond[0]
    print(f'[DEBUG] blended_cond[0]: shape={tuple(bc0.shape)}  min={bc0.min():.4f}  max={bc0.max():.4f}  mean={bc0.mean():.4f}  std={bc0.std():.4f}')

    # ── Euler schedule: from t_start → 0, linear (no rescale for sub-range) ──
    t_seq   = np.linspace(t_start, 0, steps + 1)
    t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
    print(f'[DEBUG] t_seq (first 5): {[f"{v:.4f}" for v in t_seq[:5]]}  cfg applied when t in {cfg_interval}')
    print(f'[DEBUG] slat_norm  std={std.tolist()}')
    print(f'[DEBUG] slat_norm mean={mean.tolist()}')

    print(f'Denoising {T} frames × {steps} steps  '
          f'k={k}  blend={blend}  alpha={alpha}  '
          f'cfg={cfg_strength}  interval={cfg_interval}  rescale_t={rescale_t}')

    cfg_step_count = 0
    for step_idx, (t_val, t_prev) in enumerate(tqdm(t_pairs, desc='GL denoising')):
        t_tensor = torch.tensor([1000 * t_val], device=device, dtype=torch.float32)

        # Conditional pass: standard flow_model forward with blended DINO context
        v_cond = [flow_model(all_x[i], t_tensor, blended_cond[i]) for i in range(T)]

        # Log velocity stats for frame 0 at steps 0, 12, 24
        if step_idx in (0, steps // 2, steps - 1):
            vc0 = v_cond[0].feats
            print(f'[DEBUG] step={step_idx:02d} t={t_val:.4f}  v_cond[0]: min={vc0.min():.3f}  max={vc0.max():.3f}  mean={vc0.mean():.3f}  nan={vc0.isnan().any()}')

        dt = float(t_val - t_prev)
        if cfg_interval[0] <= t_val <= cfg_interval[1]:
            cfg_step_count += 1
            v_unc = [flow_model(all_x[i], t_tensor, blended_neg_cond[i]) for i in range(T)]
            if step_idx in (0, steps // 2, steps - 1):
                vu0 = v_unc[0].feats
                vc0f = v_cond[0].feats
                cfg_v = (1 + cfg_strength) * vc0f - cfg_strength * vu0
                print(f'[DEBUG] step={step_idx:02d}  v_unc[0]: min={vu0.min():.3f}  max={vu0.max():.3f}  mean={vu0.mean():.3f}')
                print(f'[DEBUG] step={step_idx:02d}  cfg_v[0]: min={cfg_v.min():.3f}  max={cfg_v.max():.3f}  mean={cfg_v.mean():.3f}')
            new_x = [
                all_x[i].replace(
                    all_x[i].feats - dt * (
                        (1 + cfg_strength) * v_cond[i].feats
                        - cfg_strength     * v_unc[i].feats
                    )
                )
                for i in range(T)
            ]
        else:
            new_x = [
                all_x[i].replace(all_x[i].feats - dt * v_cond[i].feats)
                for i in range(T)
            ]
        all_x = new_x

        if step_idx in (0, steps // 2, steps - 1):
            xf = all_x[0].feats
            print(f'[DEBUG] step={step_idx:02d}  x[0] after update: min={xf.min():.3f}  max={xf.max():.3f}  mean={xf.mean():.3f}  nan={xf.isnan().any()}')
        torch.cuda.empty_cache()

    print(f'[DEBUG] CFG applied for {cfg_step_count}/{steps} steps')
    out_norm = torch.stack([x.feats for x in all_x], dim=0)  # (T, N_vox, 8) normalized
    print(f'[DEBUG] out_norm  min={out_norm.min():.3f}  max={out_norm.max():.3f}  mean={out_norm.mean():.3f}  std={out_norm.std():.3f}  nan={out_norm.isnan().any()}')
    print(f'[DEBUG] out_norm per-channel mean: {out_norm.mean(dim=(0,1)).tolist()}')
    out = out_norm * std + mean
    print(f'[DEBUG] out_raw   min={out.min():.3f}  max={out.max():.3f}  mean={out.mean():.3f}  std={out.std():.3f}')
    print(f'[DEBUG] out_raw  per-channel mean: {out.mean(dim=(0,1)).tolist()}')
    return out.float().cpu().numpy()


# ── Decode & render ───────────────────────────────────────────────────────────

_SCALE_WARN = False

def decode_and_render(pipeline, feats_np, coords, extr, intr):
    global _SCALE_WARN
    feats = torch.from_numpy(feats_np.astype(np.float32)).cuda()
    slat  = sp.SparseTensor(feats=feats, coords=coords)
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['gaussian'])
    g = decoded['gaussian'][0]
    raw_max = g._scaling.max().item()
    g._scaling = g._scaling.clamp(-6.0, 2.0)
    if not _SCALE_WARN and raw_max > 2.0:
        print(f'[render] _scaling clamped max={raw_max:.2f} → 2.0')
        _SCALE_WARN = True
    frames = render_utils.render_frames(
        g, extr, intr,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return frames['color'][0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_camera():
    import utils3d.torch as u3d
    fov  = torch.deg2rad(torch.tensor(40.)).cuda()
    eye  = torch.tensor([0., 0., 2.]).cuda()
    tgt  = torch.zeros(3).cuda()
    up   = torch.tensor([0., 1., 0.]).cuda()
    return [u3d.extrinsics_look_at(eye, tgt, up)], [u3d.intrinsics_from_fov_xy(fov, fov)]


def load_gt(t):
    p   = GT_VIDEO_DIR / f'frame_{t:04d}.png'
    img = Image.open(p).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float() / 255.0


def psnr(pred, gt):
    mse = float(F.mse_loss(pred.float(), gt.float()).item())
    return 10 * math.log10(1.0 / (mse + 1e-10))


def make_video(frames_dir, pattern, out, fps=15):
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
                    '-i', str(frames_dir / pattern),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)],
                   check=True, capture_output=True)


def make_comparison(gt_dir, trellis_dir, smooth_dir, out, fps=15):
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
                    '-i', str(gt_dir      / 'frame_%03d.png'),
                    '-i', str(trellis_dir / 'frame_%03d.png'),
                    '-i', str(smooth_dir  / 'frame_%03d.png'),
                    '-filter_complex', '[0][1][2]hstack=inputs=3',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)],
                   check=True, capture_output=True)


def prepare_trellis_baseline(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(1, N_FRAMES + 1):
        dst = out_dir / f'frame_{t:03d}.png'
        if dst.exists():
            continue
        src = FRAMES_150_DIR / f'frame_{t:04d}' / 'renders' / 'front.png'
        Image.open(src).convert('RGB').resize(
            (RENDER_RES, RENDER_RES), Image.LANCZOS).save(dst)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',       type=int,   default=1)
    parser.add_argument('--alpha',   type=float, default=1.0)
    parser.add_argument('--blend',   type=float, default=1.0)
    parser.add_argument('--t_start', type=float, default=1.0,
                        help='SDEdit start timestep (1.0=pure noise generation).')
    parser.add_argument('--mode',    type=str,   default='avg',
                        choices=['avg', 'temporal_spatial', 'spatial_temporal',
                                 'joint', 'all'])
    args = parser.parse_args()

    ALL_MODES = ['avg', 'temporal_spatial', 'spatial_temporal', 'joint']
    modes = ALL_MODES if args.mode == 'all' else [args.mode]

    print('Loading pipeline...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    for m in pipeline.models.values():
        for p in m.parameters():
            p.requires_grad_(False)

    # ── Per-frame conditioning: encode all 150 frames with preprocess_image ──
    print('Encoding all 150 frames with preprocess_image...')
    all_conds_np = []
    for i in range(1, N_FRAMES + 1):
        img_path = FRAMES_150_DIR / f'frame_{i:04d}' / 'renders' / 'front.png'
        img = Image.open(img_path).convert('RGB')
        img_proc = pipeline.preprocess_image(img)
        with torch.no_grad():
            cond = pipeline.encode_image([img_proc])   # (1, 1374, 1024)
        all_conds_np.append(cond.cpu().numpy())
        if i % 25 == 0 or i == 1:
            print(f'  frame {i:03d}: mean={cond.mean():.4f}  std={cond.std():.4f}')
    print(f'Done. Conditioning shape: {all_conds_np[0].shape}')

    # ── Voxel coords from SS model on frame 1 (preprocessed) ─────────────────
    print('Running SS model on frame 1 to get voxel coords...')
    T_1_tensor = torch.from_numpy(all_conds_np[0]).cuda()
    cond_ss    = {'cond': T_1_tensor, 'neg_cond': torch.zeros_like(T_1_tensor)}
    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        coords = pipeline.sample_sparse_structure(cond_ss)
    print(f'SS coords: {coords.shape[0]} voxels')
    print(f'[DEBUG] coords range: x=[{coords[:,1].min()},{coords[:,1].max()}]  y=[{coords[:,2].min()},{coords[:,2].max()}]  z=[{coords[:,3].min()},{coords[:,3].max()}]')


    extr, intr  = build_camera()
    gt_dir      = OUT_DIR / 'gt'
    trellis_dir = OUT_DIR / 'trellis_renders'
    gt_dir.mkdir(parents=True, exist_ok=True)
    prepare_trellis_baseline(trellis_dir)
    for t in range(1, N_FRAMES + 1):
        dst = gt_dir / f'frame_{t:03d}.png'
        if not dst.exists():
            gt = load_gt(t)
            Image.fromarray((gt.numpy() * 255).astype(np.uint8)).save(dst)

    # ── Load slat_cache for SDEdit init ──────────────────────────────────────
    slat_feats_np = None
    if args.t_start < 1.0:
        sc_path = REPO_ROOT / 'experiments' / 'results' / 'phase7' / 'slat_cache.npz'
        sc = np.load(sc_path)
        slat_feats_np = sc['slats']   # (150, N_vox, 8) raw SLaT features
        print(f'Loaded slat_cache: shape={slat_feats_np.shape}  t_start={args.t_start}')

    results = {}
    for mode in modes:
        tag     = f'{mode}_k{args.k}_b{args.blend}_ts{args.t_start}'
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Phase 8 GL | {tag} ===')

        slats_np = sample_all_frames_crossframe(
            pipeline, all_conds_np, coords,
            blender=BLENDERS[mode], k=args.k,
            alpha=args.alpha, blend=args.blend,
            slat_feats_np=slat_feats_np,
            t_start=args.t_start,
        )   # (150, N_vox, 8)

        # Full Gaussian decode check on frame 0
        with torch.no_grad():
            f0 = torch.from_numpy(slats_np[0].astype(np.float32)).cuda()
            s0 = sp.SparseTensor(feats=f0, coords=coords)
            dec0 = pipeline.decode_slat(s0, ['gaussian'])
            g0   = dec0['gaussian'][0]
            dc   = g0._features_dc
            print(f'[DEBUG] GENERATED frame0 slats: min={slats_np[0].min():.4f}  max={slats_np[0].max():.4f}  mean={slats_np[0].mean():.4f}  per-ch={slats_np[0].mean(axis=0).tolist()}')
            print(f'[DEBUG] GENERATED _features_dc:  shape={tuple(dc.shape)}  min={dc.min():.4f}  max={dc.max():.4f}  mean={dc.mean():.4f}')
            print(f'[DEBUG]   dc RGB means: R={dc[:,0,0].mean():.4f}  G={dc[:,0,1].mean():.4f}  B={dc[:,0,2].mean():.4f}' if dc.dim()==3 else f'[DEBUG]   dc flat mean={dc.mean():.4f}')
            print(f'[DEBUG] GENERATED _opacity:      min={g0._opacity.min():.4f}  max={g0._opacity.max():.4f}  mean={g0._opacity.mean():.4f}')
            print(f'[DEBUG] GENERATED _scaling:      min={g0._scaling.min():.4f}  max={g0._scaling.max():.4f}  mean={g0._scaling.mean():.4f}')
            print(f'[DEBUG] GENERATED _rotation:     min={g0._rotation.min():.4f}  max={g0._rotation.max():.4f}')
            print(f'[DEBUG] GENERATED xyz:            min={g0._xyz.min():.4f}  max={g0._xyz.max():.4f}  mean={g0._xyz.mean():.4f}')
            # Render frame 0 and check pixel stats
            import torch as _t
            g0c = g0; g0c._scaling = g0c._scaling.clamp(-6.0, 2.0)
            test_render = render_utils.render_frames(g0c, extr, intr,
                options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)}, verbose=False)
            pix = _t.from_numpy(test_render['color'][0]).float() / 255.0
            print(f'[DEBUG] GENERATED render frame0: shape={tuple(pix.shape)}  min={pix.min():.4f}  max={pix.max():.4f}  mean={pix.mean():.4f}')
            print(f'[DEBUG]   pixel R={pix[:,:,0].mean():.4f}  G={pix[:,:,1].mean():.4f}  B={pix[:,:,2].mean():.4f}')

        psnr_rows = []
        for t in range(N_FRAMES):
            color   = decode_and_render(pipeline, slats_np[t], coords, extr, intr)
            color_f = torch.from_numpy(color).float() / 255.0
            gt      = load_gt(t + 1)
            p       = psnr(color_f, gt)
            psnr_rows.append({'frame': t + 1, 'psnr': p})
            Image.fromarray(color).save(out_dir / f'frame_{t + 1:03d}.png')
            if (t + 1) % 25 == 0 or t == 0:
                print(f'  t={t + 1:03d}/{N_FRAMES}  PSNR={p:.2f} dB')
            torch.cuda.empty_cache()

        avg_p = np.mean([r['psnr'] for r in psnr_rows])
        results[tag] = avg_p
        print(f'{tag}  avg PSNR: {avg_p:.2f} dB')

        with open(out_dir / 'psnr.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['frame', 'psnr'])
            w.writeheader()
            w.writerows(psnr_rows)

        make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
        make_comparison(gt_dir, trellis_dir, out_dir, out_dir / 'comparison.mp4')
        print(f'Saved: {out_dir}/comparison.mp4')

    print('\n=== Summary ===')
    for name, v in results.items():
        print(f'  {name:<35}: {v:.2f} dB')
    print(f'\nOutputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
