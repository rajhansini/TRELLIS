"""
interpolation_gap_test.py

Tests whether linear DINOv2 token interpolation can capture real texture dynamics.

Pipeline:
  - Encode frame_0001 and frame_0030 front renders → T_0, T_30 (DINOv2 tokens)
  - For t in 1..30:
      alpha = t / 30
      T_t = (1-alpha)*T_0 + alpha*T_30
      z_t = flow_model(fixed_voxels_from_frame1, T_t, shared_seed)
      interp_render_t = decode(z_t) → Gaussian → TRELLIS camera render

  - GT: decode pre-encoded SLAT latents → same TRELLIS camera render
  - Actual MVAdaptor frame stored as visual reference (3rd column in video)

Modes:
  --mode a  Raw MVAdaptor front frames as conditioning (no background removal)
  --mode b  pipeline.preprocess_image() applied before encoding (rembg + crop)

Output (scripts/interp_gap_out_{MODE}/):
  error_curve.csv, error_curve.png, frames/gt_NN.png, frames/interp_NN.png,
  frames/ref_NN.png (actual video frame), comparison.mp4

Run on a GPU node (f002/f003):
  conda activate /net/projects/ranalab/rajhansini/conda_envs/trellis
  python scripts/interpolation_gap_test.py --mode a [--steps 12] [--seed 42]
  python scripts/interpolation_gap_test.py --mode b [--steps 12] [--seed 42]
"""

import os
import sys

os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import subprocess
from pathlib import Path
from PIL import Image
import csv

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent

GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs'
                     '/teapot_lava_kling_premium/teapot_lava_kling_premium_front')

SLAT_SEQ_DIR  = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'

PRETRAINED    = 'microsoft/TRELLIS-image-large'
N_FRAMES      = 30
NOISE_SEED    = 42
RENDER_RES    = 512

# MVAdaptor front camera in TRELLIS render space:
# eye=(0, 0, r) Y-up → passed via render_frames with explicit extrinsics
TRELLIS_R   = 1.5
TRELLIS_FOV = 60.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_front_image(frame_idx: int) -> Image.Image:
    path = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    return Image.open(path).convert('RGB')


def load_gt_slat(frame_idx: int, device) -> sp.SparseTensor:
    data = np.load(SLAT_SEQ_DIR / f'frame_{frame_idx:04d}' / 'latent.npz')
    coords_np = data['coords'].astype(np.int32)
    feats_np  = data['feats'].astype(np.float32)
    batch     = np.zeros((len(coords_np), 1), dtype=np.int32)
    coords4   = torch.from_numpy(np.concatenate([batch, coords_np], axis=1)).to(device)
    feats     = torch.from_numpy(feats_np).to(device)
    return sp.SparseTensor(feats=feats, coords=coords4)


def fixed_coords_from_frame1(device) -> torch.Tensor:
    data   = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    return torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)


# ---------------------------------------------------------------------------
# Rendering — MVAdaptor front camera: eye=(0,0,r), Y-up, applied to all renders
# ---------------------------------------------------------------------------
def _build_camera(device):
    # Use render_utils helper directly — handles batching/unboxing correctly
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [-1.5708], TRELLIS_R, TRELLIS_FOV   # yaw=0, pitch=-π/2 ≈ looking at +Z face
    )
    # Override with MVAdaptor front camera (eye=(0,0,r), Y-up)
    import utils3d
    eye  = torch.tensor([0.0, 0.0, TRELLIS_R], dtype=torch.float32, device=device)
    orig = torch.tensor([0.0, 0.0, 0.0],        dtype=torch.float32, device=device)
    up   = torch.tensor([0.0, 1.0, 0.0],         dtype=torch.float32, device=device)
    extr = utils3d.torch.extrinsics_look_at(eye, orig, up)   # (..., 4, 4)
    fov  = torch.deg2rad(torch.tensor(float(TRELLIS_FOV), device=device))
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)    # (..., 3, 3)
    # Strip batch dim if added by @batched decorator
    while extr.dim() > 2: extr = extr[0]
    while intr.dim() > 2: intr = intr[0]
    return [extr], [intr]


def render_gaussian(pipeline, slat: sp.SparseTensor) -> np.ndarray:
    """Decode SLAT → Gaussian → render from MVAdaptor front camera → (H,W,3) uint8."""
    with torch.no_grad():
        decoded  = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]
    device   = gaussian.get_xyz.device
    extrinsics, intrinsics = _build_camera(device)
    result = render_utils.render_frames(
        gaussian, extrinsics, intrinsics,
        {'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
    )
    return result['color'][0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(a: np.ndarray, b: np.ndarray, lpips_fn=None, device=None):
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    l2 = float(np.mean((af - bf) ** 2))
    sv = float(ssim(a, b, channel_axis=2)) if HAS_SKIMAGE else float('nan')
    lp = float('nan')
    if lpips_fn is not None and device is not None:
        def to_t(arr):
            return torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1
        with torch.no_grad():
            lp = float(lpips_fn(to_t(a), to_t(b)).item())
    return {'ssim': sv, 'l2': l2, 'lpips': lp}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    choices=['a', 'b'], default='a',
                        help='a = raw conditioning; b = preprocess_image conditioning')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--seed',    type=int, default=NOISE_SEED)
    parser.add_argument('--steps',   type=int, default=12,
                        help='SLAT sampler steps (12 = default, 6 = fast)')
    parser.add_argument('--no_gt',   action='store_true')
    args = parser.parse_args()

    out_dir    = Path(args.out_dir) if args.out_dir else \
                 REPO_ROOT / 'scripts' / f'interp_gap_out_frontcam_{args.mode}'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # -- Load pipeline --------------------------------------------------------
    print(f'Loading TRELLIS pipeline: {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device
    print(f'Pipeline on {device}  |  mode={args.mode}')

    # -- Encode anchor frames -------------------------------------------------
    def encode(frame_idx: int) -> torch.Tensor:
        img = load_front_image(frame_idx)
        if args.mode == 'b':
            print(f'  preprocessing frame {frame_idx} ...')
            img = pipeline.preprocess_image(img)
        return pipeline.encode_image([img])

    print('Encoding frame 1 (T_0) ...')
    T_0  = encode(1)
    print('Encoding frame 30 (T_30) ...')
    T_30 = encode(30)
    print(f'Token shape: {T_0.shape}')

    # Pre-encode all frames for direct (oracle) rendering
    print('Pre-encoding all 30 frames for direct oracle ...')
    T_all = {}
    for t in range(1, N_FRAMES + 1):
        T_all[t] = encode(t)
    print('Done encoding.')

    # -- Fixed voxel coords ---------------------------------------------------
    coords_fixed = fixed_coords_from_frame1(device)
    print(f'Fixed voxels: {coords_fixed.shape[0]}')

    # -- LPIPS ----------------------------------------------------------------
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
        print('LPIPS loaded')
    else:
        print('lpips not installed — using SSIM + L2 only')

    # -- Per-frame loop -------------------------------------------------------
    results = []

    for t in range(1, N_FRAMES + 1):
        alpha  = t / N_FRAMES
        T_t    = (1.0 - alpha) * T_0 + alpha * T_30
        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

        # Interpolated render
        torch.manual_seed(args.seed)
        with torch.no_grad():
            slat_interp = pipeline.sample_slat(
                cond_t, coords_fixed,
                sampler_params={'steps': args.steps},
            )
        interp_img = render_gaussian(pipeline, slat_interp)
        Image.fromarray(interp_img).save(frames_dir / f'interp_{t:02d}.png')

        # GT SLAT render (same TRELLIS camera)
        gt_img = None
        if not args.no_gt:
            slat_gt = load_gt_slat(t, device)
            gt_img  = render_gaussian(pipeline, slat_gt)
            Image.fromarray(gt_img).save(frames_dir / f'gt_{t:02d}.png')

        # Direct (oracle): flow model with actual T_t tokens — no interpolation
        cond_direct = {'cond': T_all[t], 'neg_cond': torch.zeros_like(T_all[t])}
        torch.manual_seed(args.seed)
        with torch.no_grad():
            slat_direct = pipeline.sample_slat(
                cond_direct, coords_fixed,
                sampler_params={'steps': args.steps},
            )
        direct_img = render_gaussian(pipeline, slat_direct)
        Image.fromarray(direct_img).save(frames_dir / f'direct_{t:02d}.png')

        # Reference: actual MVAdaptor frame (visual context only)
        ref_img = load_front_image(t).resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        ref_img.save(frames_dir / f'ref_{t:02d}.png')

        # Metrics: interp vs GT SLAT, and direct vs GT SLAT
        row = {'t': t, 'alpha': round(alpha, 4),
               'ssim_interp': float('nan'), 'l2_interp': float('nan'),
               'ssim_direct': float('nan'), 'l2_direct': float('nan')}
        if gt_img is not None:
            m_i = compute_metrics(interp_img, gt_img, lpips_fn, device)
            m_d = compute_metrics(direct_img, gt_img, lpips_fn, device)
            row['ssim_interp'] = m_i['ssim']
            row['l2_interp']   = m_i['l2']
            row['ssim_direct'] = m_d['ssim']
            row['l2_direct']   = m_d['l2']

        results.append(row)
        print(f't={t:02d}  α={alpha:.3f}  '
              f'interp SSIM={row["ssim_interp"]:.4f}  '
              f'direct SSIM={row["ssim_direct"]:.4f}')

    # -- Save CSV -------------------------------------------------------------
    csv_path = out_dir / 'error_curve.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'alpha',
                                               'ssim_interp', 'l2_interp',
                                               'ssim_direct', 'l2_direct'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved: {csv_path}')

    # -- Plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ts            = [r['t']            for r in results]
        ssims_interp  = [r['ssim_interp']  for r in results]
        ssims_direct  = [r['ssim_direct']  for r in results]
        l2s_interp    = [r['l2_interp']    for r in results]
        l2s_direct    = [r['l2_direct']    for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        kw = dict(linestyle='-', marker='o', markersize=4)

        axes[0].plot(ts, ssims_interp, label='interp (linear)', **kw)
        axes[0].plot(ts, ssims_direct, label='direct (oracle)', color='orange', **kw)
        axes[0].set_xlabel('Frame t')
        axes[0].set_ylabel('SSIM ↑')
        axes[0].set_title('SSIM vs GT SLAT render')
        axes[0].axvline(x=1,  color='g', linestyle='--', alpha=0.5, label='anchor t=1')
        axes[0].axvline(x=30, color='r', linestyle='--', alpha=0.5, label='anchor t=30')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, l2s_interp, label='interp (linear)', **kw)
        axes[1].plot(ts, l2s_direct, label='direct (oracle)', color='orange', **kw)
        axes[1].set_xlabel('Frame t')
        axes[1].set_ylabel('Pixel L2 ↓')
        axes[1].set_title('L2 vs GT SLAT render')
        axes[1].axvline(x=1,  color='g', linestyle='--', alpha=0.5, label='anchor t=1')
        axes[1].axvline(x=30, color='r', linestyle='--', alpha=0.5, label='anchor t=30')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        mode_label = 'raw conditioning' if args.mode == 'a' else 'preprocessed conditioning'
        plt.suptitle(f'Interpolation Gap Test — mode {args.mode.upper()} ({mode_label})',
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / 'error_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_dir}/error_curve.png')
    except Exception as e:
        print(f'Warning: plot failed ({e})')

    # -- 4-panel video: ref | gt_slat | interp | direct ----------------------
    try:
        vid_path = out_dir / 'comparison.mp4'
        ffmpeg_bin = '/usr/bin/ffmpeg'
        subprocess.run([
            ffmpeg_bin, '-y', '-framerate', '8',
            '-i', str(frames_dir / 'ref_%02d.png'),
            '-i', str(frames_dir / 'gt_%02d.png'),
            '-i', str(frames_dir / 'interp_%02d.png'),
            '-i', str(frames_dir / 'direct_%02d.png'),
            '-filter_complex', '[0:v][1:v][2:v][3:v]hstack=inputs=4',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(vid_path),
        ], check=True)
        print(f'Saved: {vid_path}')
    except Exception as e:
        print(f'Warning: video failed ({e})')

    # -- Summary --------------------------------------------------------------
    valid = [r for r in results if np.isfinite(r['ssim_interp'])]
    if valid:
        mid   = N_FRAMES // 2
        mid_r = next((r for r in valid if r['t'] == mid), None)
        print('\n=== SUMMARY ===')
        print(f'  {"t":>4}  {"interp SSIM":>12}  {"direct SSIM":>12}  {"gap":>8}')
        for r in valid:
            gap = r['ssim_direct'] - r['ssim_interp']
            print(f'  t={r["t"]:02d}  {r["ssim_interp"]:12.4f}  {r["ssim_direct"]:12.4f}  {gap:+8.4f}')
        print()
        print('  gap = direct - interp  (positive = interpolation worse than oracle)')

    print(f'\nOutputs: {out_dir}')


if __name__ == '__main__':
    main()
