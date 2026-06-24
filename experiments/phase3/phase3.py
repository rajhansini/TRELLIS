"""
Phase 3 — Extrapolation Beyond Frame 150
==========================================
Extend the linear token interpolation direction past frame 150.

For frame t > 150:
  alpha  = (t - 1) / 149          (> 1.0 for extrapolation)
  T_t    = T_0 + alpha*(T_150 - T_0)   = (1-alpha)*T_0 + alpha*T_150
  voxels = same fixed coords as Phase 0
  seed   = same fixed seed

Produces:
  - extra_NNN.png renders for frames 151..150+N_EXTRA
  - full.mp4  — Phase 0 (frames 1-150) + Phase 3 extrapolation stitched
  - extra.mp4 — extrapolation frames only

Pass --reverse to also generate frames 0..-(N_EXTRA-1) (backward extrapolation).

Run (A40 required):
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase3/phase3.py [--n_extra 50] [--steps 25] [--seed 42]
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PHASE0_DIR   = REPO_ROOT / 'experiments' / 'results' / 'phase0'
OUT_DIR      = REPO_ROOT / 'experiments' / 'results' / 'phase3'
SLAT_SEQ_DIR = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED   = 'microsoft/TRELLIS-image-large'

N_FRAMES_P0 = 150
NOISE_SEED  = 42
RENDER_RES  = 512
STEPS       = 25


# ─────────────────────────────────────────────────────────────────────────────

def fixed_coords(device):
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


def render_gaussian(pipeline, slat):
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [0.0], 2.0, 40.0,
    )
    frames = render_utils.render_frames(
        gaussian, extr, intr,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return frames['color'][0]


def make_video(frame_pattern, out_path, fps=15):
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frame_pattern),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
    ], check=True)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_extra',  type=int,  default=50,
                        help='Number of frames to extrapolate past frame 150')
    parser.add_argument('--steps',    type=int,  default=STEPS)
    parser.add_argument('--seed',     type=int,  default=NOISE_SEED)
    parser.add_argument('--reverse',  action='store_true',
                        help='Also extrapolate backward (before frame 1)')
    parser.add_argument('--start',    type=int,  default=1,
                        help='Resume from this extra frame index (1-based)')
    args = parser.parse_args()

    frames_dir = OUT_DIR / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print(f'Loading TRELLIS: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device
    print(f'Device: {device}  steps={args.steps}  seed={args.seed}  n_extra={args.n_extra}')

    # ── Load anchor tokens ────────────────────────────────────────────────────
    tokens_path = PHASE0_DIR / 'tokens.npz'
    if not tokens_path.exists():
        raise FileNotFoundError(f'tokens.npz not found at {tokens_path}. Run Phase 0 first.')
    data  = np.load(tokens_path)
    T_0   = torch.from_numpy(data['T_0'].astype(np.float32)).to(device)
    T_150 = torch.from_numpy(data['T_150'].astype(np.float32)).to(device)
    print(f'Anchor tokens loaded. Shape: {T_0.shape}')

    coords_fixed = fixed_coords(device)
    print(f'Fixed voxels: {coords_fixed.shape[0]}')

    # Direction vector (unnormalized — same scale as interpolation)
    delta = T_150 - T_0   # (1, 1374, 1024)

    # ── Forward extrapolation (frames 151 → 150+n_extra) ─────────────────────
    print(f'\n--- Forward extrapolation: frames 151 to {N_FRAMES_P0 + args.n_extra} ---')
    for i in range(args.start, args.n_extra + 1):
        t     = N_FRAMES_P0 + i               # absolute frame number (151, 152, ...)
        alpha = (t - 1) / (N_FRAMES_P0 - 1)   # > 1.0
        T_t   = T_0 + alpha * delta            # extrapolated token

        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}
        torch.manual_seed(args.seed)
        with torch.no_grad():
            slat = pipeline.sample_slat(
                cond_t, coords_fixed,
                sampler_params={'steps': args.steps},
            )

        img = render_gaussian(pipeline, slat)
        Image.fromarray(img).save(frames_dir / f'extra_fwd_{t:04d}.png')
        print(f'  t={t}  α={alpha:.4f}')

    # ── Reverse extrapolation (frames 0 → -(n_extra-1)) ──────────────────────
    if args.reverse:
        print(f'\n--- Reverse extrapolation: frames 0 to -{args.n_extra - 1} ---')
        for i in range(1, args.n_extra + 1):
            t     = 1 - i                      # 0, -1, -2, ...
            alpha = (t - 1) / (N_FRAMES_P0 - 1)   # < 0.0
            T_t   = T_0 + alpha * delta

            cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}
            torch.manual_seed(args.seed)
            with torch.no_grad():
                slat = pipeline.sample_slat(
                    cond_t, coords_fixed,
                    sampler_params={'steps': args.steps},
                )

            img = render_gaussian(pipeline, slat)
            # save with padded index so 1=first reverse frame
            Image.fromarray(img).save(frames_dir / f'extra_rev_{i:04d}.png')
            print(f'  t={t}  α={alpha:.4f}')

    # ── Stitch full video: Phase 0 + extrapolation ────────────────────────────
    print('\nStitching full video (Phase 0 + extrapolation) ...')
    p0_frames = sorted((PHASE0_DIR / 'frames').glob('trellis_*.png'))
    extra_frames = sorted(frames_dir.glob('extra_fwd_*.png'))

    # symlink / copy into a single numbered sequence
    seq_dir = OUT_DIR / 'full_seq'
    seq_dir.mkdir(exist_ok=True)
    all_frames = p0_frames + extra_frames
    for idx, src in enumerate(all_frames, start=1):
        dst = seq_dir / f'frame_{idx:04d}.png'
        if not dst.exists():
            dst.symlink_to(src)

    try:
        make_video(seq_dir / 'frame_%04d.png', OUT_DIR / 'full.mp4', fps=15)
        print(f'Saved: {OUT_DIR}/full.mp4')
    except Exception as e:
        print(f'full.mp4 failed: {e}')

    try:
        make_video(frames_dir / 'extra_fwd_%04d.png', OUT_DIR / 'extra.mp4', fps=15)
        print(f'Saved: {OUT_DIR}/extra.mp4')
    except Exception as e:
        print(f'extra.mp4 failed: {e}')

    print(f'\nDone. Outputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
