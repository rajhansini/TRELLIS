"""
Compute PSNR, flicker, and FID for:
  1. Raw TRELLIS fixed noise   (diag_full_comparison/raw_trellis)
  2. Raw TRELLIS per-frame noise (diag_three_videos/raw_trellis_perframe)
  3. MCFM v2_C, v2_D, v3_C, v3_D (already exist)
Compare all to GT.
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image

N_FRAMES = 150
ENH = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')
GT_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
              '/outputs/teapot_lava_kling_premium'
              '/teapot_lava_kling_premium_front/all_frames_150')

DIRS = {
    'raw_fixed'   : ENH / 'diag_full_comparison' / 'raw_trellis',
    'raw_perframe': ENH / 'diag_three_videos'     / 'raw_trellis_perframe',
    'v2_C'        : ENH / 'results_mcfm_v2_C_seed6_fixednoise' / 'beta0p0',
    'v2_D'        : ENH / 'results_mcfm_v2_D_seed6_fixednoise' / 'beta0p0',
    'v3_C'        : ENH / 'results_mcfm_v3_C_seed6_fixednoise' / 'beta0p0',
    'v3_D'        : ENH / 'results_mcfm_v3_D_seed6_fixednoise' / 'beta0p0',
}

LOG = ENH / 'diag_three_videos' / 'metrics.log'

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def load_frame(path, size=518):
    return np.array(Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)).astype(np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def flicker(frames):
    diffs = [np.mean(np.abs(frames[i+1] - frames[i])) for i in range(len(frames)-1)]
    return np.mean(diffs)


log('=== compute_metrics_raw ===')

# Load GT frames once
log('\nLoading GT frames...')
gt_frames = []
for fi in range(1, N_FRAMES + 1):
    gt_frames.append(load_frame(GT_DIR / f'frame_{fi:04d}.png'))
log(f'  GT loaded: {len(gt_frames)} frames')

# Compute GT flicker as reference
gt_flicker = flicker(gt_frames)
log(f'  GT flicker: {gt_flicker:.5f}')

results = {}

for name, d in DIRS.items():
    log(f'\n[{name}] {d}')
    frames = []
    psnr_vals = []
    for fi in range(1, N_FRAMES + 1):
        p = d / f'frame_{fi:04d}.png'
        if not p.exists():
            log(f'  MISSING: {p}')
            continue
        f = load_frame(p)
        frames.append(f)
        psnr_vals.append(psnr(f, gt_frames[fi-1]))

    if len(frames) < N_FRAMES:
        log(f'  Only {len(frames)}/{N_FRAMES} frames found — skipping.')
        continue

    avg_psnr  = float(np.mean(psnr_vals))
    std_psnr  = float(np.std(psnr_vals))
    flk       = float(flicker(frames))

    results[name] = {'psnr': avg_psnr, 'psnr_std': std_psnr, 'flicker': flk, 'n': len(frames)}
    log(f'  PSNR={avg_psnr:.3f} dB  std={std_psnr:.3f}  flicker={flk:.5f}')

log('\n=== SUMMARY ===')
log(f'{"Name":<16}  {"PSNR":>8}  {"std":>6}  {"Flicker":>9}')
log(f'{"GT":16}  {"—":>8}  {"—":>6}  {gt_flicker:9.5f}')
for name, r in results.items():
    log(f'{name:<16}  {r["psnr"]:8.3f}  {r["psnr_std"]:6.3f}  {r["flicker"]:9.5f}')

out = ENH / 'diag_three_videos' / 'metrics.json'
with open(out, 'w') as f:
    json.dump({'gt_flicker': gt_flicker, 'results': results}, f, indent=2)
log(f'\nSaved: {out}')
