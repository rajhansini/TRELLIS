"""
Phase 5 — Direct SLAT Optimization Per Frame
==============================================
For each frame t, optimize the SLAT feature tensor z_t directly via gradient
descent against the pixel-level GT front render — no training, no module.

Pipeline (all differentiable):
  z_t  (learnable)
   ↓  pipeline.decode_slat    (frozen VAE decoder, grad flows through)
  Gaussian splats
   ↓  GaussianRenderer        (diff_gaussian_rasterization)
  pixel tensor (3, H, W)
   ↓  MSE + LPIPS loss vs GT
  backward → update z_t

Initialization:  z_t ← flow_model(lerp(T_0, T_150, alpha))  [same as Phase 0]
Optimization:    Adam, n_steps iterations per frame

Outputs (experiments/results/phase5/):
  frames/phase5_NNN.png   — optimized render
  frames/phase0_NNN.png   — Phase 0 init render (for comparison)
  frames/gt_NNN.png       — GT front.png
  comparison.mp4          — 3-panel: GT | Phase 0 | Phase 5
  loss_curve.csv          — per-frame final loss + PSNR
  run.log

Run (A40 required):
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase5/phase5.py [--n_steps 100] [--lr 5e-3] [--seed 42]
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

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
PHASE0_DIR     = REPO_ROOT / 'experiments' / 'results' / 'phase0'
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase5'
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
SLAT_SEQ_DIR   = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
NOISE_SEED = 42
RENDER_RES = 512
STEPS      = 25
N_OPT      = 100    # optimization steps per frame
LR         = 5e-3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fixed_coords(device):
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


def load_gt(frame_idx, device):
    """Load GT front.png as (3, H, W) float32 tensor in [0,1]."""
    path = FRAMES_150_DIR / f'frame_{frame_idx:04d}' / 'renders' / 'front.png'
    img  = Image.open(path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.).permute(2, 0, 1).to(device)


def build_camera(device):
    """yaw=0, pitch=0, r=2.0, fov=40 — confirmed matching GT orientation."""
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [0.0], 2.0, 40.0,
    )
    return extr[0].to(device), intr[0].to(device)


def render_differentiable(renderer, gaussian, extr, intr):
    """Returns (3, H, W) float32 tensor with gradient graph intact."""
    return renderer.render(gaussian, extr, intr)['color']


def psnr(pred, gt):
    mse = float(F.mse_loss(pred, gt).item())
    return 10 * math.log10(1.0 / (mse + 1e-10))


def to_uint8(tensor):
    """(3, H, W) float → (H, W, 3) uint8."""
    return (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int,   default=N_OPT,
                        help='Optimization steps per frame')
    parser.add_argument('--lr',      type=float, default=LR)
    parser.add_argument('--seed',    type=int,   default=NOISE_SEED)
    parser.add_argument('--steps',   type=int,   default=STEPS,
                        help='Flow model diffusion steps for init')
    parser.add_argument('--start',   type=int,   default=1,
                        help='Resume from this frame (1-based)')
    parser.add_argument('--log_every', type=int, default=20,
                        help='Print loss every N opt steps')
    args = parser.parse_args()

    frames_dir = OUT_DIR / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print(f'Loading TRELLIS: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device
    print(f'Device: {device}  n_steps={args.n_steps}  lr={args.lr}')

    # Freeze all pipeline weights
    for p in pipeline.parameters():
        p.requires_grad_(False)

    # ── Anchor tokens ─────────────────────────────────────────────────────────
    tokens_path = PHASE0_DIR / 'tokens.npz'
    if not tokens_path.exists():
        raise FileNotFoundError(f'Run Phase 0 first: {tokens_path}')
    data  = np.load(tokens_path)
    T_0   = torch.from_numpy(data['T_0'].astype(np.float32)).to(device)
    T_150 = torch.from_numpy(data['T_150'].astype(np.float32)).to(device)
    print(f'Tokens loaded: {T_0.shape}')

    coords_fixed = fixed_coords(device)
    print(f'Fixed voxels: {coords_fixed.shape[0]}')

    # ── Renderer (created once, reused) ───────────────────────────────────────
    renderer = GaussianRenderer(rendering_options={
        'resolution': RENDER_RES,
        'bg_color': (1.0, 1.0, 1.0),
    })

    extr, intr = build_camera(device)

    # ── Per-frame optimization ─────────────────────────────────────────────────
    rows = []

    for t in range(args.start, N_FRAMES + 1):
        alpha  = (t - 1) / (N_FRAMES - 1)
        T_t    = (1.0 - alpha) * T_0 + alpha * T_150
        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

        gt = load_gt(t, device)   # (3, H, W) float32

        # ── Step 1: Phase 0 init (flow model, no grad) ────────────────────────
        torch.manual_seed(args.seed)
        with torch.no_grad():
            slat_init = pipeline.sample_slat(
                cond_t, coords_fixed,
                sampler_params={'steps': args.steps},
            )

        # Save Phase 0 init render for comparison
        p0_img = to_uint8(render_differentiable(
            renderer,
            pipeline.decode_slat(slat_init, ['gaussian'])['gaussian'][0],
            extr, intr,
        ))
        Image.fromarray(p0_img).save(frames_dir / f'phase0_{t:03d}.png')

        # ── Step 2: Make SLAT feats learnable ─────────────────────────────────
        z = slat_init.feats.detach().float().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=args.lr)

        init_loss = None
        final_loss = None

        # ── Step 3: Optimization loop ──────────────────────────────────────────
        for step in range(args.n_steps):
            optimizer.zero_grad()

            slat_opt = sp.SparseTensor(feats=z, coords=slat_init.coords)
            decoded  = pipeline.decode_slat(slat_opt, ['gaussian'])
            gaussian = decoded['gaussian'][0]

            color = render_differentiable(renderer, gaussian, extr, intr)  # (3,H,W)
            loss  = F.mse_loss(color, gt)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if step == 0:
                init_loss = loss_val
            if (step + 1) % args.log_every == 0 or step == args.n_steps - 1:
                print(f'  t={t:03d}  step={step+1:03d}/{args.n_steps}'
                      f'  loss={loss_val:.6f}  PSNR={psnr(color, gt):.2f} dB')
            final_loss = loss_val

        # ── Step 4: Save optimized render ─────────────────────────────────────
        with torch.no_grad():
            slat_final = sp.SparseTensor(feats=z.detach(), coords=slat_init.coords)
            decoded    = pipeline.decode_slat(slat_final, ['gaussian'])
            final_color = render_differentiable(
                renderer, decoded['gaussian'][0], extr, intr,
            )

        opt_img = to_uint8(final_color)
        gt_img  = to_uint8(gt)
        Image.fromarray(opt_img).save(frames_dir / f'phase5_{t:03d}.png')
        Image.fromarray(gt_img).save(frames_dir / f'gt_{t:03d}.png')

        final_psnr = psnr(final_color, gt)
        init_psnr  = psnr(
            torch.from_numpy(p0_img.astype(np.float32) / 255.).permute(2, 0, 1).to(device),
            gt,
        )
        rows.append({'t': t, 'alpha': round(alpha, 4),
                     'init_loss': round(float(init_loss), 6),
                     'final_loss': round(float(final_loss), 6),
                     'init_psnr': round(init_psnr, 3),
                     'final_psnr': round(final_psnr, 3)})
        print(f't={t:03d}  α={alpha:.3f}  '
              f'PSNR: {init_psnr:.2f} → {final_psnr:.2f} dB  '
              f'(Δ={final_psnr - init_psnr:+.2f})')

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / 'loss_curve.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['t', 'alpha', 'init_loss', 'final_loss', 'init_psnr', 'final_psnr'])
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved: {csv_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ts = [r['t'] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ts, [r['init_psnr']  for r in rows], label='Phase 0 init', alpha=0.7)
        axes[0].plot(ts, [r['final_psnr'] for r in rows], label='Phase 5 opt',  alpha=0.9)
        axes[0].set_xlabel('Frame'); axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR: Phase 0 init vs Phase 5 optimized')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ts, [r['final_psnr'] - r['init_psnr'] for r in rows], color='green')
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('Frame'); axes[1].set_ylabel('ΔPSNR (dB)')
        axes[1].set_title(f'PSNR gain from {args.n_steps}-step optimization')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'psnr_curve.png', dpi=150)
        plt.close()
        print(f'Saved: {OUT_DIR}/psnr_curve.png')
    except Exception as e:
        print(f'Plot failed: {e}')

    # ── 3-panel comparison video: GT | Phase 0 | Phase 5 ─────────────────────
    try:
        vid_path = OUT_DIR / 'comparison.mp4'
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '15',
            '-i', str(frames_dir / 'gt_%03d.png'),
            '-i', str(frames_dir / 'phase0_%03d.png'),
            '-i', str(frames_dir / 'phase5_%03d.png'),
            '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid_path),
        ], check=True)
        print(f'Saved: {vid_path}')
    except Exception as e:
        print(f'Video failed: {e}')

    avg_gain = np.mean([r['final_psnr'] - r['init_psnr'] for r in rows])
    print(f'\n=== Summary ===')
    print(f'  Avg PSNR gain over {N_FRAMES} frames: {avg_gain:+.2f} dB')
    print(f'  n_steps={args.n_steps}  lr={args.lr}')
    if avg_gain > 1.0:
        print('  → Optimization improves renders significantly.')
        print('    SLAT space is expressive enough to fit the GT texture dynamics.')
    else:
        print('  → Minimal gain. Either LR/steps need tuning, or the bottleneck')
        print('    is the fixed geometry (coords), not the SLAT features.')
    print(f'\nOutputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
