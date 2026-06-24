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

# TRELLIS default camera — matches the training distribution for SH evaluation
TRELLIS_R     = 2.0
TRELLIS_FOV   = 40.0
TRELLIS_PITCH = 0.25
TRELLIS_YAW   = 0.0


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
# Rendering — TRELLIS default camera (Z-up, pitch=0.25, r=2, fov=40)
# ---------------------------------------------------------------------------
def render_gaussian(pipeline, slat: sp.SparseTensor) -> np.ndarray:
    """Decode SLAT → Gaussian → render with TRELLIS default camera → (H,W,3) uint8."""
    with torch.no_grad():
        decoded  = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]
    # num_frames=1 → yaw=0, pitch=0.25 (first frame of the orbit)
    frames   = render_utils.render_video(
        gaussian, num_frames=1,
        r=TRELLIS_R, fov=TRELLIS_FOV,
        resolution=RENDER_RES, bg_color=(1, 1, 1),
    )
    return frames['color'][0]  # (H, W, 3) uint8


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
                 REPO_ROOT / 'scripts' / f'interp_gap_out_{args.mode}'
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

        # Reference: actual MVAdaptor frame (visual context only)
        ref_img = load_front_image(t).resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        ref_img.save(frames_dir / f'ref_{t:02d}.png')

        # Metrics (interp vs GT SLAT render — apples-to-apples)
        row = {'t': t, 'alpha': round(alpha, 4),
               'ssim': float('nan'), 'l2': float('nan'), 'lpips': float('nan')}
        if gt_img is not None:
            m = compute_metrics(interp_img, gt_img, lpips_fn, device)
            row.update(m)

        results.append(row)
        print(f't={t:02d}  α={alpha:.3f}  '
              f'SSIM={row["ssim"]:.4f}  L2={row["l2"]:.6f}  LPIPS={row["lpips"]:.4f}')

    # -- Save CSV -------------------------------------------------------------
    csv_path = out_dir / 'error_curve.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'alpha', 'ssim', 'l2', 'lpips'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved: {csv_path}')

    # -- Plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ts     = [r['t']    for r in results]
        ssims  = [r['ssim'] for r in results]
        l2s    = [r['l2']   for r in results]
        lpipss = [r['lpips'] for r in results]

        has_lpips_data = any(np.isfinite(v) for v in lpipss)
        ncols = 3 if has_lpips_data else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
        kw = dict(linestyle='-', marker='o', markersize=4)

        for ax, vals, label, ylabel, better in [
            (axes[0], ssims, 'SSIM',  'SSIM ↑',    'higher'),
            (axes[1], l2s,   'L2',    'Pixel L2 ↓', 'lower'),
        ]:
            ax.plot(ts, vals, **kw)
            ax.set_xlabel('Frame t')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{label}: interp vs GT SLAT  ({better} = better)')
            ax.axvline(x=1,  color='g', linestyle='--', alpha=0.5, label='anchor t=1')
            ax.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='anchor t=30')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        if has_lpips_data:
            axes[2].plot(ts, lpipss, color='purple', **kw)
            axes[2].set_xlabel('Frame t')
            axes[2].set_ylabel('LPIPS ↓')
            axes[2].set_title('LPIPS: interp vs GT SLAT  (lower = better)')
            axes[2].axvline(x=1,  color='g', linestyle='--', alpha=0.5)
            axes[2].axvline(x=30, color='r', linestyle='--', alpha=0.5)
            axes[2].grid(True, alpha=0.3)

        mode_label = 'raw conditioning' if args.mode == 'a' else 'preprocessed conditioning'
        plt.suptitle(f'Interpolation Gap Test — mode {args.mode.upper()} ({mode_label})',
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / 'error_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_dir}/error_curve.png')
    except Exception as e:
        print(f'Warning: plot failed ({e})')

    # -- 3-panel video: ref | gt_slat | interp --------------------------------
    try:
        vid_path = out_dir / 'comparison.mp4'
        subprocess.run([
            'ffmpeg', '-y', '-framerate', '8',
            '-i', str(frames_dir / 'ref_%02d.png'),
            '-i', str(frames_dir / 'gt_%02d.png'),
            '-i', str(frames_dir / 'interp_%02d.png'),
            '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(vid_path),
        ], check=True, capture_output=True)
        print(f'Saved: {vid_path}')
    except Exception as e:
        print(f'Warning: video failed ({e})')

    # -- Summary --------------------------------------------------------------
    valid = [r for r in results if np.isfinite(r['ssim'])]
    if valid:
        best  = max(valid, key=lambda r: r['ssim'])
        worst = min(valid, key=lambda r: r['ssim'])
        mid   = N_FRAMES // 2
        mid_r = next((r for r in valid if r['t'] == mid), None)
        print('\n=== SUMMARY ===')
        print(f'  Best  SSIM: t={best["t"]:02d}  {best["ssim"]:.4f}')
        print(f'  Worst SSIM: t={worst["t"]:02d}  {worst["ssim"]:.4f}')
        if mid_r:
            print(f'  Mid   SSIM: t={mid_r["t"]:02d}  {mid_r["ssim"]:.4f}'
                  f'   (anchors: t=1 → {valid[0]["ssim"]:.4f}, t=30 → {valid[-1]["ssim"]:.4f})')
        print()
        print('  If mid SSIM << anchor SSIM: interpolation misses real dynamics.')
        print('  If mid SSIM ≈ anchor SSIM:  interpolation already captures it.')

    print(f'\nOutputs: {out_dir}')


if __name__ == '__main__':
    main()
