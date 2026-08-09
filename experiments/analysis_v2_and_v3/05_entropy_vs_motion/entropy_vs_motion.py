"""
05 — Attention entropy vs GT motion magnitude.

Reads attn_weights.npy (150, n_window) from step08 for each mode.
Reads GT video frames to compute per-frame motion magnitude.
Computes Shannon entropy of attention weights per frame.

Hypothesis: high-motion frames → high entropy (model is uncertain, blends more)
            low-motion frames  → low entropy  (model confident, sharp focus on current frame)

Pearson correlation is the quantitative paper claim.

NO GPU required.

Output: 05_entropy_vs_motion/results/

Usage:
  python entropy_vs_motion.py
  python entropy_vs_motion.py --mode v2_C
"""

import sys, argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from scipy.stats import pearsonr as _pearsonr
    def pearsonr(x, y):
        r, p = _pearsonr(x, y)
        return float(r), float(p)
except ImportError:
    def pearsonr(x, y):
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float('nan')

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent.parent
_ENH      = _ROOT / 'experiments' / 'enhancement'
VIDEO_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                 '/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
N_FRAMES   = 150
RENDER_RES = 518

COLORS = {
    'v2_C': '#61afef',
    'v2_D': '#56b6c2',
    'v3_C': '#e06c75',
    'v3_D': '#d19a66',
}


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def compute_gt_motion() -> np.ndarray:
    """Per-frame GT motion: motion[t] = mean |frame_t - frame_{t-1}|, shape (N_FRAMES,)."""
    print('[MOTION] Computing GT motion over 150 frames...')
    motion = np.zeros(N_FRAMES, dtype=np.float32)
    prev   = None
    for i in range(1, N_FRAMES + 1):
        img = np.array(
            Image.open(VIDEO_DIR / f'frame_{i:04d}.png').convert('RGB')
                  .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        ).astype(np.float32) / 255.0
        if prev is not None:
            motion[i - 1] = float(np.abs(img - prev).mean())
        prev = img
    motion[0] = motion[1]  # replicate second frame for boundary
    print(f'  motion: mean={motion.mean():.5f}  max={motion.max():.5f}  '
          f'min={motion.min():.5f}')
    return motion


def compute_entropy(attn: np.ndarray) -> np.ndarray:
    """Shannon entropy per frame: H(w) = -sum(w * log(w+eps)), shape (N,)."""
    return -np.sum(attn * np.log(attn + 1e-9), axis=1)


def analyze_mode(mode: str, motion: np.ndarray, out_dir: Path) -> dict:
    attn_path = _ENH / f'results_attn_{mode}' / 'attn_weights.npy'
    if not attn_path.exists():
        print(f'  [{mode}] MISSING: {attn_path}  (run step08 for this mode first)')
        return {}

    attn    = np.load(attn_path)      # (150, n_win)
    entropy = compute_entropy(attn)   # (150,)
    n_win   = attn.shape[1]
    max_H   = float(np.log(n_win))   # max possible entropy (uniform dist)

    r, p = pearsonr(motion[1:], entropy[1:])
    p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'
    print(f'  [{mode}] n_win={n_win}  entropy_mean={entropy.mean():.4f}  '
          f'max_H={max_H:.4f}  r(motion,H)={r:.4f}  p={p_str}')

    frame_idx = np.arange(1, N_FRAMES + 1)
    col       = COLORS.get(mode, '#abb2bf')
    win_labels = ['t', 't+1'] if mode.endswith('_C') else ['t-1', 't', 't+1']
    wcolors    = ['#e06c75', '#61afef', '#98c379']

    # ── 3-panel figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle(f'Attention entropy vs GT motion — {mode}  '
                 f'Pearson r={r:.3f}  (p={p_str})', fontsize=12)

    # Panel 1: dual-axis time series
    ax_m   = axes[0]
    ax_h   = ax_m.twinx()
    ax_m.plot(frame_idx, motion, color='#98c379', lw=1.2, alpha=0.85, label='GT motion')
    ax_h.plot(frame_idx, entropy, color=col,       lw=1.2, alpha=0.90, label='Entropy')
    ax_m.set_xlabel('Frame index')
    ax_m.set_ylabel('GT motion  (mean |Δframe|)', color='#98c379', fontsize=9)
    ax_h.set_ylabel('Attention entropy (nats)',   color=col,        fontsize=9)
    ax_m.set_title('Time series')
    ax_m.grid(True, alpha=0.3)
    lh1, ll1 = ax_m.get_legend_handles_labels()
    lh2, ll2 = ax_h.get_legend_handles_labels()
    ax_m.legend(lh1 + lh2, ll1 + ll2, fontsize=8, loc='upper right')

    # Panel 2: scatter
    axes[1].scatter(motion[1:], entropy[1:], c=col, alpha=0.45, s=14, edgecolors='none')
    xf = np.linspace(motion[1:].min(), motion[1:].max(), 100)
    m_fit, b_fit = np.polyfit(motion[1:], entropy[1:], 1)
    axes[1].plot(xf, m_fit * xf + b_fit, 'k--', lw=1.5, alpha=0.65,
                 label=f'Linear fit  r={r:.3f}')
    axes[1].set_xlabel('GT motion magnitude')
    axes[1].set_ylabel('Attention entropy (nats)')
    axes[1].set_title('Scatter: motion vs entropy')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: attention weights over frames
    for w in range(n_win):
        axes[2].plot(frame_idx, attn[:, w], color=wcolors[w], lw=1.1,
                     alpha=0.85, label=f'w_{win_labels[w]}')
    axes[2].axhline(1.0 / n_win, color='gray', ls='--', lw=0.8, alpha=0.55, label='uniform')
    axes[2].set_xlabel('Frame index')
    axes[2].set_ylabel('Softmax attention weight')
    axes[2].set_title('Attention weights over frames')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-0.03, 1.06)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / f'entropy_vs_motion_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [{mode}] saved entropy_vs_motion_{mode}.png')

    # CSV
    with open(out_dir / f'entropy_vs_motion_{mode}.csv', 'w') as f:
        f.write('frame,motion,entropy\n')
        for i in range(N_FRAMES):
            f.write(f'{i + 1},{motion[i]:.6f},{entropy[i]:.6f}\n')

    return {
        'mode': mode, 'r': r, 'p': p,
        'mean_entropy': float(entropy.mean()),
        'max_H': max_H,
        'frac_max': float(entropy.mean() / max_H),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None,
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args  = parser.parse_args()
    modes = [args.mode] if args.mode else ['v2_C', 'v2_D', 'v3_C', 'v3_D']

    out_dir = _HERE / 'results' / 'entropy_vs_motion'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'entropy_vs_motion.log')
    sys.stderr = sys.stdout

    motion = compute_gt_motion()

    results = []
    for mode in modes:
        r = analyze_mode(mode, motion, out_dir)
        if r:
            results.append(r)

    if results:
        print('\n[SUMMARY]')
        hdr = f"{'Mode':<8} {'r':<8} {'p':<10} {'mean_H':<10} {'frac_max_H':<12}"
        print(hdr); print('-' * len(hdr))
        for r in results:
            p_str = f"{r['p']:.4f}" if not np.isnan(r['p']) else 'N/A'
            print(f"  {r['mode']:<8} {r['r']:<8.4f} {p_str:<10} "
                  f"{r['mean_entropy']:<10.4f} {r['frac_max']:<12.4f}")

    print(f'\n[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
