"""
Phase 8 — Temporal Smoothing on Colored Renders
================================================
Applies V1/V2/V3 temporal smoothing directly to the 150 per-frame colored
renders from trellis_150_frames/frame_XXXX/renders/front.png.

No GPU required. Output: experiments/results/phase8_render/<mode>_k<k>/

Run:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  python experiments/phase8/phase8_render_smooth.py --mode all --k 1
"""

import argparse, csv, math, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
GT_VIDEO_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                      '/outputs/teapot_lava_kling_premium'
                      '/teapot_lava_kling_premium_front/all_frames_150')
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase8_render'
N_FRAMES       = 150
RENDER_RES     = 512


# ── Load all 150 colored renders into a tensor ───────────────────────────────

def load_renders():
    frames = []
    for t in range(1, N_FRAMES + 1):
        src = FRAMES_150_DIR / f'frame_{t:04d}' / 'renders' / 'front.png'
        img = Image.open(src).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        frames.append(torch.from_numpy(np.array(img)).float() / 255.0)  # (H, W, 3)
    data = torch.stack(frames, dim=0)  # (T, H, W, 3)
    print(f'Loaded renders: {data.shape}  min={data.min():.3f}  max={data.max():.3f}')
    return data


def load_gt(t):
    path = GT_VIDEO_DIR / f'frame_{t:04d}.png'
    img = Image.open(path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float() / 255.0


# ── Smoothing modes ────────────────────────────────────────────────────────────
# Each function takes data (T, H, W, 3), frame index t, window half-size k
# and returns smoothed frame (H, W, 3) float in [0,1].

def smooth_avg(data, t, k):
    """V1: simple average over window."""
    lo, hi = max(0, t - k), min(N_FRAMES - 1, t + k)
    return data[lo:hi + 1].mean(dim=0)


def smooth_weighted(data, t, k, alpha=2.0):
    """V2: weight current frame alpha× more than neighbors."""
    lo, hi = max(0, t - k), min(N_FRAMES - 1, t + k)
    weights = torch.ones(hi - lo + 1)
    cur_idx = t - lo
    weights[cur_idx] = alpha
    weights = weights / weights.sum()
    window = data[lo:hi + 1]           # (W, H, W, 3)
    return (window * weights[:, None, None, None]).sum(dim=0)


def smooth_gauss(data, t, k):
    """V3: Gaussian-weighted window (sigma = k/2)."""
    lo, hi = max(0, t - k), min(N_FRAMES - 1, t + k)
    idxs   = torch.arange(lo, hi + 1, dtype=torch.float)
    sigma  = max(k / 2.0, 0.5)
    weights = torch.exp(-0.5 * ((idxs - t) / sigma) ** 2)
    weights = weights / weights.sum()
    window = data[lo:hi + 1]
    return (window * weights[:, None, None, None]).sum(dim=0)


MODES = {
    'avg':      smooth_avg,
    'weighted': smooth_weighted,
    'gauss':    smooth_gauss,
}


# ── PSNR ──────────────────────────────────────────────────────────────────────

def psnr(pred, gt):
    mse = float(F.mse_loss(pred, gt).item())
    return 10 * math.log10(1.0 / (mse + 1e-10))


# ── Video helpers ──────────────────────────────────────────────────────────────

def make_video(frames_dir, pattern, out_path, fps=15):
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frames_dir / pattern),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
    ], check=True, capture_output=True)


def make_comparison_video(gt_dir, orig_dir, smoothed_dir, out_path, fps=15):
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(gt_dir      / 'frame_%03d.png'),
        '-i', str(orig_dir    / 'frame_%03d.png'),
        '-i', str(smoothed_dir / 'frame_%03d.png'),
        '-filter_complex', '[0][1][2]hstack=inputs=3',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
    ], check=True, capture_output=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_mode(data, mode_fn, mode_name, k, out_dir, gt_dir, orig_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    psnr_rows = []
    for t in range(N_FRAMES):
        smoothed = mode_fn(data, t, k).clamp(0, 1)  # (H, W, 3)
        img_u8   = (smoothed.numpy() * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(out_dir / f'frame_{t + 1:03d}.png')

        gt = load_gt(t + 1)
        p  = psnr(smoothed, gt)
        psnr_rows.append({'frame': t + 1, 'psnr': p})

        if (t + 1) % 25 == 0 or t == 0:
            print(f'  t={t + 1:03d}/{N_FRAMES}  PSNR={p:.2f} dB')

    avg_p = np.mean([r['psnr'] for r in psnr_rows])
    print(f'{mode_name}_k{k}  avg PSNR: {avg_p:.2f} dB')

    with open(out_dir / 'psnr.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'psnr'])
        w.writeheader(); w.writerows(psnr_rows)

    make_video(out_dir, 'frame_%03d.png', out_dir / 'render.mp4')
    make_comparison_video(gt_dir, orig_dir, out_dir, out_dir / 'comparison.mp4')
    print(f'Saved: {out_dir}/comparison.mp4')
    return avg_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',    type=int, default=1)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['avg', 'weighted', 'gauss', 'all'])
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == 'all' else [args.mode]

    print('Loading 150 colored renders...')
    data = load_renders()  # (150, H, W, 3)

    # Prepare GT and original (unsmoothed) dirs
    gt_dir   = OUT_DIR / 'gt'
    orig_dir = OUT_DIR / 'original'
    gt_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    print('Copying GT and original renders...')
    for t in range(1, N_FRAMES + 1):
        gt_dst = gt_dir / f'frame_{t:03d}.png'
        if not gt_dst.exists():
            gt = load_gt(t)
            Image.fromarray((gt.numpy() * 255).astype(np.uint8)).save(gt_dst)

        orig_dst = orig_dir / f'frame_{t:03d}.png'
        if not orig_dst.exists():
            orig_u8 = (data[t - 1].numpy() * 255).astype(np.uint8)
            Image.fromarray(orig_u8).save(orig_dst)

    results = {}
    for mode in modes_to_run:
        tag     = f'{mode}_k{args.k}'
        out_dir = OUT_DIR / tag
        print(f'\n=== {tag} ===')
        avg_p = run_mode(data, MODES[mode], mode, args.k, out_dir, gt_dir, orig_dir)
        results[tag] = avg_p

    print('\n=== Summary ===')
    for name, v in results.items():
        print(f'  {name:<30}: {v:.2f} dB')
    print(f'\nOutputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
