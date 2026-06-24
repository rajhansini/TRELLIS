"""
Phase 0 — Linear Token Interpolation Baseline
==============================================
Goal: Can G_L's cross-attention accept blended T_0/T_150 tokens to reproduce
      a 150-frame dynamic texture sequence, with no training?

Pipeline per frame t (1..150):
  alpha  = (t-1) / 149          # 0.0 at t=1, 1.0 at t=150
  T_t    = (1-alpha)*T_0 + alpha*T_150
  voxels = fixed coords from frame_0001 SLAT  (geometry frozen)
  seed   = fixed (reproducible across frames)
  SLAT_t = flow_model(T_t, voxels, seed, steps=STEPS)
  render_t = decode(SLAT_t) → Gaussian → TRELLIS default camera

GT reference: frame_XXXX/renders/front.png from 150-frame MVAdaptor dataset.

Output (experiments/results/phase0/):
  frames/trellis_NNN.png  — TRELLIS interpolated render
  frames/gt_NNN.png       — MVAdaptor front.png GT frame
  comparison.mp4          — side-by-side GT | TRELLIS interp
  tokens.npz              — saved T_0, T_150 (for Phase 1)
  run.log

Run on f002/f003 (A40 GPU required):
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \\
      python experiments/phase0/phase0.py [--steps 25] [--seed 42]
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import torch
import subprocess
from pathlib import Path
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
SLAT_SEQ_DIR   = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
NOISE_SEED = 42
RENDER_RES = 512
STEPS      = 25   # default; override with --steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gt_front_path(frame_idx: int) -> Path:
    return FRAMES_150_DIR / f'frame_{frame_idx:04d}' / 'renders' / 'front.png'


def encode_input_path(frame_idx: int) -> Path:
    return FRAMES_150_DIR / f'frame_{frame_idx:04d}' / 'renders' / 'front.png'


def load_gt_frame(frame_idx: int) -> np.ndarray:
    img = Image.open(gt_front_path(frame_idx)).convert('RGB')
    return np.array(img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS))


def fixed_coords_from_frame1(device) -> torch.Tensor:
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


def render_gaussian(pipeline, slat: sp.SparseTensor) -> np.ndarray:
    """Decode SLAT → Gaussian → render at yaw=0, pitch=0, r=2.0, fov=40 → (H,W,3) uint8.

    Confirmed by camera probe: this camera matches GT front.png orientation.
    """
    with torch.no_grad():
        decoded  = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [0.0], 2.0, 40.0,
    )
    frames = render_utils.render_frames(
        gaussian, extrinsics, intrinsics,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return frames['color'][0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',   type=int,  default=STEPS)
    parser.add_argument('--seed',    type=int,  default=NOISE_SEED)
    parser.add_argument('--preprocess', action='store_true',
                        help='Run pipeline.preprocess_image() before encoding')
    parser.add_argument('--start',   type=int,  default=1,
                        help='Start frame index (for resuming)')
    args = parser.parse_args()

    out_dir    = Path(__file__).parent.parent / 'results' / 'phase0'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # -- Load pipeline -------------------------------------------------------
    print(f'Loading TRELLIS pipeline: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device
    print(f'Pipeline on {device}  |  steps={args.steps}  seed={args.seed}  '
          f'preprocess={args.preprocess}')

    # -- Encode anchor frames ------------------------------------------------
    def encode(frame_idx: int) -> torch.Tensor:
        img = Image.open(encode_input_path(frame_idx)).convert('RGB')
        if args.preprocess:
            img = pipeline.preprocess_image(img)
        return pipeline.encode_image([img])

    print('Encoding T_0 (frame 1) ...')
    T_0   = encode(1)
    print('Encoding T_150 (frame 150) ...')
    T_150 = encode(150)
    print(f'Token shape: {T_0.shape}')

    # -- Fixed voxel coords --------------------------------------------------
    coords_fixed = fixed_coords_from_frame1(device)
    print(f'Fixed voxels: {coords_fixed.shape[0]}')

    # -- Save tokens for Phase 1 ---------------------------------------------
    tokens_path = out_dir / 'tokens.npz'
    np.savez(tokens_path,
             T_0=T_0.cpu().float().numpy(),
             T_150=T_150.cpu().float().numpy())
    print(f'Saved anchor tokens: {tokens_path}')

    # -- Per-frame loop ------------------------------------------------------
    for t in range(args.start, N_FRAMES + 1):
        alpha  = (t - 1) / (N_FRAMES - 1)   # 0.0 at t=1, 1.0 at t=150
        T_t    = (1.0 - alpha) * T_0 + alpha * T_150
        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

        torch.manual_seed(args.seed)
        with torch.no_grad():
            slat = pipeline.sample_slat(
                cond_t, coords_fixed,
                sampler_params={'steps': args.steps},
            )

        trellis_img = render_gaussian(pipeline, slat)
        Image.fromarray(trellis_img).save(frames_dir / f'trellis_{t:03d}.png')

        gt_img = load_gt_frame(t)
        Image.fromarray(gt_img).save(frames_dir / f'gt_{t:03d}.png')

        print(f't={t:03d}/{N_FRAMES}  α={alpha:.4f}')

    # -- Video ---------------------------------------------------------------
    print('Making comparison video ...')
    vid_path = out_dir / 'comparison.mp4'
    try:
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '15',
            '-i', str(frames_dir / 'gt_%03d.png'),
            '-i', str(frames_dir / 'trellis_%03d.png'),
            '-filter_complex', '[0:v][1:v]hstack=inputs=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(vid_path),
        ], check=True)
        print(f'Saved: {vid_path}')
    except Exception as e:
        print(f'Video failed: {e}')
        print('Run manually:')
        print(f'  cd {frames_dir} && /usr/bin/ffmpeg -y -framerate 15 '
              f'-i gt_%03d.png -i trellis_%03d.png '
              f'-filter_complex "[0:v][1:v]hstack=inputs=2" '
              f'-c:v libx264 -pix_fmt yuv420p {vid_path}')

    print(f'\nDone. Outputs: {out_dir}')


if __name__ == '__main__':
    main()
