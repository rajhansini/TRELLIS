"""
Step 09 — MCFM Attention Entropy + Motion Correlation Analysis

Reads attn_weights.npy from step08 results (no GPU, no re-running MCFM).

For each frame i:
  entropy(i) = -sum(w * log(w + eps))   over window positions
  motion(i)  = mean |frame_i - frame_{i-1}| pixel difference

Hypothesis: high entropy → uniform attention across frames → more smoothing
            low entropy  → peaked on current frame → sharper, less smoothing
            If hypothesis holds: entropy correlates with motion content.

BACKWARD COMPATIBLE: does not touch step08, mcfm.py, or any training scripts.

Results in results_attn_{mode}/ (same dir as step08, new files):
  entropy_{mode}.npy       — (150,) entropy per frame
  motion_{mode}.npy        — (150,) motion per frame
  entropy_plot_{mode}.png  — entropy + motion dual-axis plot over 150 frames
  scatter_{mode}.png       — scatter: entropy vs motion with Pearson r
  entropy_analysis.log     — full log

Usage:
  python step09_attn_entropy_analysis.py --mode v2_C
  python step09_attn_entropy_analysis.py --mode v2_D
  python step09_attn_entropy_analysis.py --mode v3_C
  python step09_attn_entropy_analysis.py --mode v3_D
"""

import sys, os, argparse as _ap
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent.parent

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode', type=str, default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_PRE, _ = _pre.parse_known_args()

_RESULTS = _HERE / f'results_attn_{_PRE.mode}'
_RESULTS.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

sys.stdout = _Tee(_RESULTS / 'entropy_analysis.log')
sys.stderr = sys.stdout

import json, time
import numpy as np
from PIL import Image
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
N_FRAMES      = 150
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
EPS = 1e-10


def compute_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """attn_weights: (N, n_window) → entropy: (N,)"""
    return -(attn_weights * np.log(attn_weights + EPS)).sum(axis=1)


def load_frame_gray(frame_idx: int) -> np.ndarray:
    """Load GT frame as float32 grayscale [0,1]."""
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def compute_motion(n_frames: int) -> np.ndarray:
    """frame-to-frame mean absolute pixel difference, shape (n_frames,).
    frame 1 gets motion=0 (no previous frame)."""
    motion = np.zeros(n_frames, dtype=np.float32)
    prev = load_frame_gray(1)
    for i in range(2, n_frames + 1):
        curr = load_frame_gray(i)
        motion[i - 1] = np.abs(curr - prev).mean()
        prev = curr
        if i % 30 == 0:
            print(f'  motion computed up to frame {i}')
    return motion


def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode', type=str, default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args  = parser.parse_args()
    mode  = args.mode

    if mode.endswith('_C'):
        labels   = ['current(t)', 'next(t+1)']
    else:
        labels   = ['prev(t-1)', 'current(t)', 'next(t+1)']

    print('=' * 72)
    print('Step 09 — MCFM Attention Entropy + Motion Correlation')
    print('=' * 72)
    print(f'  mode    : {mode}')
    print(f'  results : {_RESULTS}')
    print()

    # ── Load attn weights from step08 ─────────────────────────────────────────
    npy_path = _RESULTS / 'attn_weights.npy'
    if not npy_path.exists():
        raise FileNotFoundError(
            f'{npy_path} not found — run step08_attn_weight_analysis.py --mode {mode} first')

    attn_weights = np.load(npy_path)   # (150, n_window)
    print(f'[LOAD] attn_weights.npy: shape={attn_weights.shape}')
    print(f'  mean per window position:')
    for j, lbl in enumerate(labels):
        print(f'    {lbl}: {attn_weights[:, j].mean():.4f}')

    # ── Entropy ───────────────────────────────────────────────────────────────
    entropy = compute_entropy(attn_weights)   # (150,)
    np.save(_RESULTS / f'entropy_{mode}.npy', entropy)
    print(f'\n[ENTROPY] shape={entropy.shape}')
    print(f'  mean={entropy.mean():.4f}  min={entropy.min():.4f}  max={entropy.max():.4f}')
    print(f'  max-entropy uniform = {-np.log(1.0/attn_weights.shape[1]):.4f}')

    # ── Motion ────────────────────────────────────────────────────────────────
    motion_path = _RESULTS / f'motion_{mode}.npy'
    # Motion is independent of mode — reuse if already computed
    shared_motion = _HERE / 'results_attn_v2_C' / 'motion_v2_C.npy'
    if motion_path.exists():
        motion = np.load(motion_path)
        print(f'\n[MOTION] Loaded existing {motion_path.name}')
    elif shared_motion.exists() and mode != 'v2_C':
        motion = np.load(shared_motion)
        np.save(motion_path, motion)
        print(f'\n[MOTION] Copied from v2_C (motion is mode-independent)')
    else:
        print(f'\n[MOTION] Computing frame-to-frame pixel diff...')
        motion = compute_motion(N_FRAMES)
        np.save(motion_path, motion)
    print(f'  mean={motion.mean():.4f}  min={motion.min():.4f}  max={motion.max():.4f}')

    # ── Correlation ───────────────────────────────────────────────────────────
    # Skip frame 1 (motion=0, no prior frame)
    r, p = stats.pearsonr(entropy[1:], motion[1:])
    print(f'\n[CORRELATION] Pearson r={r:.4f}  p={p:.4e}')
    if p < 0.05:
        print(f'  Significant correlation (p<0.05)')
        if r > 0:
            print(f'  → Higher motion = higher entropy = more smoothing')
        else:
            print(f'  → Higher motion = lower entropy = less smoothing (sharper current frame)')
    else:
        print(f'  No significant correlation (p>0.05)')

    frame_axis = np.arange(1, N_FRAMES + 1)

    # ── Plot 1: entropy + motion over frames ──────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.plot(frame_axis, entropy, color='#4C9BE8', linewidth=1.5, label='entropy')
    ax2.plot(frame_axis, motion,  color='#E07B54', linewidth=1.5, label='motion', alpha=0.8)

    ax1.set_xlabel('Frame index', fontsize=12)
    ax1.set_ylabel('Attention entropy', color='#4C9BE8', fontsize=11)
    ax2.set_ylabel('Motion (mean |Δpixel|)', color='#E07B54', fontsize=11)
    ax1.set_xlim(1, N_FRAMES)
    ax1.set_title(f'MCFM {mode} — entropy vs motion  (Pearson r={r:.3f}, p={p:.2e})', fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_RESULTS / f'entropy_plot_{mode}.png', dpi=150)
    plt.close(fig)
    print(f'\n[PLOT] entropy_plot_{mode}.png saved')

    # ── Plot 2: scatter entropy vs motion ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(motion[1:], entropy[1:], alpha=0.5, s=20, color='#4C9BE8')
    m, b = np.polyfit(motion[1:], entropy[1:], 1)
    x_line = np.linspace(motion[1:].min(), motion[1:].max(), 100)
    ax.plot(x_line, m * x_line + b, color='#E07B54', linewidth=2, label=f'fit (r={r:.3f})')
    ax.set_xlabel('Motion (mean |Δpixel|)', fontsize=12)
    ax.set_ylabel('Attention entropy', fontsize=12)
    ax.set_title(f'MCFM {mode} — entropy vs motion scatter', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_RESULTS / f'scatter_{mode}.png', dpi=150)
    plt.close(fig)
    print(f'[PLOT] scatter_{mode}.png saved')

    print(f'\n[DONE] Results in {_RESULTS}')


if __name__ == '__main__':
    main()
