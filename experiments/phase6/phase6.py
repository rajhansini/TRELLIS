"""
Phase 6 — Flow-Field Velocity Extrapolation
=============================================
G_L is a flow model: at each ODE step it computes a velocity v_θ(z, t, T)
telling z which direction to move. Instead of guessing alpha and re-running
the full ODE for each extrapolated frame, we use the model's own velocity
field to get the direction FOR FREE in one forward pass.

Two strategies implemented (select with --mode):

  A) SLAT-diff  (mode=slat_diff, default):
     Compute z_0 = G_L(T_0) and z_150 = G_L(T_150) once.
     SLAT-space velocity: v = (z_150 - z_0) / 149   [feats per frame]
     Extrapolate by stepping in that direction:
       z_t = z_150 + (t-150) * v     (no new flow model call per frame)

  B) Flow-field query  (mode=flow_field):
     Add tiny noise to z_150 (sigma=0.1) → z_noisy.
     Query the model ONCE: v_flow = v_θ(z_noisy, sigma, T_150)
     This is the velocity field evaluated at frame 150 in SLAT space —
     the model's literal "which way to move" vector at this point.
     Extrapolate: z_t = z_150 + (t-150) * step * v_flow_norm

Mode A is purer (uses the actual inter-frame displacement in SLAT space).
Mode B is what the user describes literally — "query the velocity field direction."

Both are compared against Phase 3 (token-space linear extrapolation, full ODE).

Output: experiments/results/phase6/
  frames/vN_TXXX.png   — rendered frame t, mode N (A or B)
  comparison.mp4       — GT(if avail) | Phase3 | Phase6-A | Phase6-B
  run.log

Run (A40 required):
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase6/phase6.py [--mode both] [--n_extra 50]
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

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
PHASE0_DIR     = REPO_ROOT / 'experiments' / 'results' / 'phase0'
PHASE3_DIR     = REPO_ROOT / 'experiments' / 'results' / 'phase3'
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase6'
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
SLAT_SEQ_DIR   = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
NOISE_SEED = 42
RENDER_RES = 512
STEPS      = 25


# ─────────────────────────────────────────────────────────────────────────────

def fixed_coords(device):
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


def render_slat(pipeline, slat):
    """Decode SLAT → Gaussian → render at confirmed front camera → uint8 (H,W,3)."""
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


def run_flow_model(pipeline, cond_tokens, coords_fixed, seed):
    """Run full ODE → returns de-normalized SLAT (same as sample_slat output)."""
    cond_t = {'cond': cond_tokens, 'neg_cond': torch.zeros_like(cond_tokens)}
    torch.manual_seed(seed)
    with torch.no_grad():
        return pipeline.sample_slat(cond_t, coords_fixed, sampler_params={'steps': STEPS})


def to_normalized(slat, pipeline):
    """De-normalized SLAT feats → normalized (flow-model space)."""
    std  = torch.tensor(pipeline.slat_normalization['std'])[None].to(slat.feats.device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(slat.feats.device)
    return (slat.feats - mean) / std


def from_normalized(feats_norm, pipeline, device):
    """Normalized feats → de-normalized."""
    std  = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return feats_norm * std + mean


def make_video(frame_dir, pattern, out_path, fps=15):
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frame_dir / pattern),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
    ], check=True)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',     choices=['slat_diff', 'flow_field', 'both'],
                        default='both')
    parser.add_argument('--n_extra',  type=int,   default=50,
                        help='Frames to generate beyond frame 150')
    parser.add_argument('--seed',     type=int,   default=NOISE_SEED)
    parser.add_argument('--sigma',    type=float, default=0.1,
                        help='Noise level for flow-field query (mode=flow_field)')
    parser.add_argument('--step_scale', type=float, default=1.0,
                        help='Scale for per-frame SLAT step size')
    args = parser.parse_args()

    frames_dir = OUT_DIR / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print(f'Loading TRELLIS: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device

    # ── Load anchor tokens ────────────────────────────────────────────────────
    tokens_path = PHASE0_DIR / 'tokens.npz'
    if not tokens_path.exists():
        raise FileNotFoundError(f'Run Phase 0 first: {tokens_path}')
    data  = np.load(tokens_path)
    T_0   = torch.from_numpy(data['T_0'].astype(np.float32)).to(device)
    T_150 = torch.from_numpy(data['T_150'].astype(np.float32)).to(device)
    token_delta = T_150 - T_0

    coords_fixed = fixed_coords(device)
    print(f'Voxels: {coords_fixed.shape[0]}  |  n_extra={args.n_extra}  mode={args.mode}')

    # ── Step 1: Generate anchor SLATs (needed for both modes) ─────────────────
    print('\n[1/3] Generating anchor SLATs (T_0 and T_150) ...')
    z_0   = run_flow_model(pipeline, T_0,   coords_fixed, args.seed)
    z_150 = run_flow_model(pipeline, T_150, coords_fixed, args.seed)
    print(f'  z_0   feats: {z_0.feats.shape}   dtype: {z_0.feats.dtype}')
    print(f'  z_150 feats: {z_150.feats.shape}  dtype: {z_150.feats.dtype}')

    # ── Mode A: SLAT-diff velocity ────────────────────────────────────────────
    if args.mode in ('slat_diff', 'both'):
        print('\n[2/3] Mode A — SLAT-diff velocity extrapolation')
        # Per-frame velocity in de-normalized SLAT space
        v_slat = (z_150.feats - z_0.feats) / (N_FRAMES - 1)   # (N_vox, C)
        print(f'  v_slat norm: {v_slat.norm().item():.4f}  per frame')

        for i in range(1, args.n_extra + 1):
            t          = N_FRAMES + i
            n_frames_extra = i   # how many frames past frame 150
            z_t_feats  = z_150.feats + n_frames_extra * args.step_scale * v_slat
            z_t        = sp.SparseTensor(feats=z_t_feats, coords=z_150.coords)

            img = render_slat(pipeline, z_t)
            Image.fromarray(img).save(frames_dir / f'vA_{t:04d}.png')
            print(f'  A  t={t}  Δ_frames={n_frames_extra}')

    # ── Mode B: Flow-field velocity query ─────────────────────────────────────
    if args.mode in ('flow_field', 'both'):
        print('\n[2/3] Mode B — Flow-field velocity query at frame 150')
        flow_model = pipeline.models['slat_flow_model']

        # Normalize z_150 to flow-model space, add noise, query velocity field
        z_150_norm = to_normalized(z_150, pipeline)
        torch.manual_seed(args.seed)
        eps = torch.randn_like(z_150_norm)

        # Forward noising: z_sigma = (1-sigma)*z_clean + sigma*noise
        sigma = args.sigma
        z_noisy_feats = (1 - sigma) * z_150_norm + sigma * eps
        z_noisy = sp.SparseTensor(feats=z_noisy_feats, coords=z_150.coords)

        t_tensor = torch.tensor([1000 * sigma] * 1,
                                 device=device, dtype=torch.float32)

        print(f'  Querying v_θ(z_150_noisy, σ={sigma}, T_150) ...')
        with torch.no_grad():
            v_flow_raw = flow_model(z_noisy, t_tensor, T_150)

        # v_flow_raw is in normalized SLAT space; convert direction to de-norm space
        if isinstance(v_flow_raw, sp.SparseTensor):
            v_flow_feats = v_flow_raw.feats   # (N_vox, C) normalized
        else:
            v_flow_feats = v_flow_raw

        # De-normalize velocity direction (scale only, no mean shift for direction)
        std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
        v_flow_denorm = v_flow_feats * std   # direction in de-normalized space

        v_norm = v_flow_denorm.norm()
        print(f'  v_flow norm (de-norm space): {v_norm.item():.4f}')

        # Scale: how big a step per frame?
        # Reference: SLAT-diff gives ||z_150 - z_0|| / 149 per frame
        v_slat_ref = (z_150.feats - z_0.feats) / (N_FRAMES - 1)
        ref_norm = v_slat_ref.norm()
        # Scale v_flow to have same magnitude as slat-diff velocity
        scale = (ref_norm / (v_norm + 1e-8)) * args.step_scale
        print(f'  Calibrated scale (matching slat-diff magnitude): {scale.item():.6f}')

        for i in range(1, args.n_extra + 1):
            t = N_FRAMES + i
            z_t_feats = z_150.feats + i * scale * v_flow_denorm
            z_t = sp.SparseTensor(feats=z_t_feats, coords=z_150.coords)

            img = render_slat(pipeline, z_t)
            Image.fromarray(img).save(frames_dir / f'vB_{t:04d}.png')
            print(f'  B  t={t}  Δ_frames={i}')

    # ── Step 3: Stitch comparison video ───────────────────────────────────────
    print('\n[3/3] Building comparison video ...')

    # Load Phase 3 extrapolation frames if available
    p3_frames = sorted(PHASE3_DIR.glob('frames/extra_fwd_*.png'))
    p0_frames = sorted((PHASE0_DIR / 'frames').glob('trellis_*.png'))   # 1-150

    # Build a full-length sequence for each mode
    def stitch_with_p0(extra_pattern, label):
        seq_dir = OUT_DIR / f'seq_{label}'
        seq_dir.mkdir(exist_ok=True)
        all_src = p0_frames + sorted(frames_dir.glob(extra_pattern))
        for idx, src in enumerate(all_src, start=1):
            dst = seq_dir / f'frame_{idx:04d}.png'
            if not dst.exists():
                dst.symlink_to(src.resolve())
        return seq_dir

    try:
        if args.mode in ('slat_diff', 'both'):
            seq = stitch_with_p0('vA_*.png', 'A')
            make_video(seq, 'frame_%04d.png', OUT_DIR / 'full_slat_diff.mp4')
            print(f'  Saved: {OUT_DIR}/full_slat_diff.mp4')
            make_video(frames_dir, 'vA_%04d.png', OUT_DIR / 'extra_slat_diff.mp4')
            print(f'  Saved: {OUT_DIR}/extra_slat_diff.mp4')

        if args.mode in ('flow_field', 'both'):
            seq = stitch_with_p0('vB_*.png', 'B')
            make_video(seq, 'frame_%04d.png', OUT_DIR / 'full_flow_field.mp4')
            print(f'  Saved: {OUT_DIR}/full_flow_field.mp4')
            make_video(frames_dir, 'vB_%04d.png', OUT_DIR / 'extra_flow_field.mp4')
            print(f'  Saved: {OUT_DIR}/extra_flow_field.mp4')
    except Exception as e:
        print(f'  Video failed: {e}')

    print(f'\nDone. Outputs: {OUT_DIR}')
    print()
    print('Key outputs:')
    print('  full_slat_diff.mp4  — Phase 0 (1-150) + SLAT-diff extrapolation (151+)')
    print('  full_flow_field.mp4 — Phase 0 (1-150) + flow-field extrapolation (151+)')
    print('  extra_slat_diff.mp4 — SLAT-diff extrapolation only')
    print('  extra_flow_field.mp4— flow-field extrapolation only')
    print()
    print('If slat_diff stays smooth → SLAT space is the right interpolation space.')
    print('If flow_field drifts faster/slower → velocity field direction is informative.')


if __name__ == '__main__':
    main()
