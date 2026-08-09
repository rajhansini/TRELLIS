"""
02 — Beta enhancement sanity check.

Reads existing rendered frames from results_mcfm_{mode}_seed6_fixednoise/beta{X}p0/
to verify the enhancement mechanism behaves as expected:
  - High beta → current frame dominates → more flicker
  - Low / zero beta → uniform weights → more temporal smoothing

Metrics computed from existing renders (NO GPU required):
  1. Temporal flicker  : mean |frame_{t+1} - frame_t| over all t, all pixels
  2. PSNR vs GT        : mean PSNR over 150 frames
  3. Std of per-frame mean brightness (proxy for temporal variance)

Output: results/beta_sanity/

Usage:
  python beta_sanity.py --mode v2_C
  python beta_sanity.py               # runs v2_C, v2_D, v3_C, v3_D
"""

import sys, argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent.parent
_ENH      = _ROOT / 'experiments' / 'enhancement'
VIDEO_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                 '/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
N_FRAMES  = 150
RENDER_RES = 518


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def load_frames(frames_dir: Path) -> np.ndarray:
    """Load all PNG frames → (N, H, W, 3) float32 [0,1]."""
    frames = []
    for i in range(1, N_FRAMES + 1):
        p = frames_dir / f'frame_{i:04d}.png'
        if not p.exists():
            return None
        img = Image.open(p).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        frames.append(np.array(img).astype(np.float32) / 255.0)
    return np.stack(frames)  # (N, H, W, 3)


def temporal_flicker(frames: np.ndarray) -> float:
    """Mean absolute frame-to-frame difference."""
    diff = np.abs(frames[1:] - frames[:-1])
    return float(diff.mean())


def mean_psnr(rendered: np.ndarray, gt: np.ndarray) -> float:
    mse_per = ((rendered - gt) ** 2).mean(axis=(1, 2, 3))
    psnr_per = 10 * np.log10(1.0 / (mse_per + 1e-8))
    return float(psnr_per.mean())


def analyze_mode(mode: str, gt_frames: np.ndarray, out_dir: Path):
    base = _ENH / f'results_mcfm_{mode}_seed6_fixednoise'
    beta_dirs = sorted(base.glob('beta*p*'))
    if not beta_dirs:
        print(f'  [{mode}] No beta dirs found in {base}')
        return

    results = {}
    for bd in beta_dirs:
        frames = load_frames(bd)
        if frames is None:
            print(f'  [{mode}] {bd.name}: incomplete, skipping')
            continue
        flicker = temporal_flicker(frames)
        psnr    = mean_psnr(frames, gt_frames)
        # parse beta value from folder name: beta0p0 → 0.0, beta4p0 → 4.0
        raw = bd.name.replace('beta', '').replace('p', '.')
        try:
            beta_val = float(raw)
        except ValueError:
            continue
        results[beta_val] = {'flicker': flicker, 'psnr': psnr}
        print(f'  [{mode}] beta={beta_val:5.1f}  flicker={flicker:.5f}  psnr={psnr:.2f}dB')

    if not results:
        return

    betas   = sorted(results)
    flickers = [results[b]['flicker'] for b in betas]
    psnrs    = [results[b]['psnr']    for b in betas]

    # Save CSV
    csv_path = out_dir / f'beta_sanity_{mode}.csv'
    with open(csv_path, 'w') as f:
        f.write('beta,flicker,psnr\n')
        for b in betas:
            f.write(f"{b},{results[b]['flicker']:.6f},{results[b]['psnr']:.4f}\n")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Beta sanity check — {mode}', fontsize=13)

    ax1.plot(betas, flickers, 'o-', color='#e06c75', lw=2, ms=7)
    ax1.set_xlabel('Beta (enhancement factor)')
    ax1.set_ylabel('Temporal flicker  (mean |Δframe|)')
    ax1.set_title('Flicker vs Beta\n(↑ = more flickering = less smoothing)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.6, label='β=0 (no enh.)')
    ax1.legend(fontsize=8)

    ax2.plot(betas, psnrs, 's-', color='#61afef', lw=2, ms=7)
    ax2.set_xlabel('Beta (enhancement factor)')
    ax2.set_ylabel('PSNR vs GT (dB)')
    ax2.set_title('PSNR vs Beta')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / f'beta_sanity_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [{mode}] saved beta_sanity_{mode}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None,
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args  = parser.parse_args()
    modes = [args.mode] if args.mode else ['v2_C', 'v2_D', 'v3_C', 'v3_D']

    out_dir = _HERE / 'results' / 'beta_sanity'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'beta_sanity.log')
    sys.stderr = sys.stdout

    print('[BETA SANITY] Loading GT frames...')
    gt_frames = load_frames(VIDEO_DIR)
    if gt_frames is None:
        print(f'ERROR: GT frames not found at {VIDEO_DIR}')
        return
    print(f'  GT frames: {gt_frames.shape}')

    for mode in modes:
        print(f'\n[{mode}]')
        analyze_mode(mode, gt_frames, out_dir)

    print(f'\n[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
