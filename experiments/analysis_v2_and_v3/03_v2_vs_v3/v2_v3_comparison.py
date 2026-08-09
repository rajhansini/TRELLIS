"""
03 — v2 vs v3 attention weight comparison.

Reads attn_weights.npy (150, n_window) saved by step08 for all 4 modes.
Produces:
  1. Bar chart: mean triplet [w_{t-1}, w_t, w_{t+1}] per mode
  2. Line plot: current frame weight w_t over 150 frames, v2_C vs v3_C and v2_D vs v3_D
  3. Violin / distribution plot of attention weights across all frames per mode

NO GPU required.

Usage:
  python v2_v3_comparison.py
"""

import sys, argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_ENH  = _ROOT / 'experiments' / 'enhancement'
N_FRAMES = 150


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


COLORS = {
    'v2_C': '#61afef',
    'v2_D': '#56b6c2',
    'v3_C': '#e06c75',
    'v3_D': '#d19a66',
}

WIN_LABELS = {
    'C': ['t', 't+1'],
    'D': ['t-1', 't', 't+1'],
}


def load_attn(mode: str):
    p = _ENH / f'results_attn_{mode}' / 'attn_weights.npy'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p} — run step08 first')
    return np.load(p)  # (150, n_window)


def main():
    out_dir = _HERE / 'results' / 'v2_vs_v3'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'comparison.log')
    sys.stderr = sys.stdout

    print('[V2 vs V3] Loading attention weights from step08...')
    data = {}
    for mode in ['v2_C', 'v2_D', 'v3_C', 'v3_D']:
        data[mode] = load_attn(mode)
        print(f'  {mode}: shape={data[mode].shape}  '
              f'mean={data[mode].mean(axis=0).round(4)}')

    # ── Plot 1: Mean attention weights bar chart ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Mean attention weights: v2 vs v3', fontsize=13)

    for ax, suffix, modes in [
        (axes[0], 'C (2-frame window)', ['v2_C', 'v3_C']),
        (axes[1], 'D (3-frame window)', ['v2_D', 'v3_D']),
    ]:
        labels = WIN_LABELS[suffix[0]]
        x      = np.arange(len(labels))
        w      = 0.35
        for k, (mode, offset) in enumerate(zip(modes, [-w/2, w/2])):
            means = data[mode].mean(axis=0)
            bars  = ax.bar(x + offset, means, w, label=mode,
                           color=COLORS[mode], alpha=0.85)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('Mean softmax attention weight')
        ax.set_title(f'Window {suffix}')
        ax.legend(); ax.set_ylim(0, 1.0)
        ax.axhline(1.0 / len(labels), color='gray', ls='--', lw=0.8, alpha=0.5,
                   label='uniform')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(out_dir / 'mean_attn_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[PLOT] mean_attn_bar.png')

    # ── Plot 2: Current frame weight over 150 frames ──────────────────────────
    frame_idx = np.arange(1, N_FRAMES + 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('Current-frame attention weight w_t over 150 frames', fontsize=13)

    for ax, (m1, m2) in zip(axes, [('v2_C', 'v3_C'), ('v2_D', 'v3_D')]):
        # current frame index: 0 for C (first slot), 1 for D (middle slot)
        cur_idx = 0 if m1.endswith('_C') else 1
        w2 = data[m1][:, cur_idx]
        w3 = data[m2][:, cur_idx]
        ax.plot(frame_idx, w2, color=COLORS[m1], lw=1.2, label=m1, alpha=0.9)
        ax.plot(frame_idx, w3, color=COLORS[m2], lw=1.2, label=m2, alpha=0.9)
        n_win = data[m1].shape[1]
        ax.axhline(1.0 / n_win, color='gray', ls='--', lw=0.8, alpha=0.5, label='uniform')
        ax.set_ylabel('w_t (current frame)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame index')
    plt.tight_layout()
    fig.savefig(out_dir / 'current_frame_weight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[PLOT] current_frame_weight.png')

    # ── Plot 3: Distribution violin ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Distribution of current-frame weight w_t across 150 frames', fontsize=13)

    for ax, suffix, modes in [
        (axes[0], 'C', ['v2_C', 'v3_C']),
        (axes[1], 'D', ['v2_D', 'v3_D']),
    ]:
        cur_idx = 0 if suffix == 'C' else 1
        vdata   = [data[m][:, cur_idx] for m in modes]
        vp      = ax.violinplot(vdata, positions=[1, 2], showmedians=True)
        for body, mode in zip(vp['bodies'], modes):
            body.set_facecolor(COLORS[mode])
            body.set_alpha(0.7)
        ax.set_xticks([1, 2]); ax.set_xticklabels(modes)
        ax.set_ylabel('w_t (current frame weight)')
        ax.set_title(f'Window {suffix}')
        n_win = 2 if suffix == 'C' else 3
        ax.axhline(1.0 / n_win, color='gray', ls='--', lw=0.8, alpha=0.5, label='uniform')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(out_dir / 'wt_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[PLOT] wt_distribution.png')

    # ── Summary stats ─────────────────────────────────────────────────────────
    print('\n[SUMMARY]')
    print(f"{'Mode':<8} {'n_win':<6} {'mean_t':<10} {'std_t':<10} "
          f"{'mean_t-1':<10} {'mean_t+1':<10}")
    for mode in ['v2_C', 'v3_C', 'v2_D', 'v3_D']:
        d      = data[mode]
        n_win  = d.shape[1]
        cur    = 0 if mode.endswith('_C') else 1
        mt     = d[:, cur].mean()
        st     = d[:, cur].std()
        m_prev = d[:, cur - 1].mean() if cur > 0 else float('nan')
        m_next = d[:, cur + 1].mean() if cur + 1 < n_win else float('nan')
        print(f"  {mode:<8} {n_win:<6} {mt:<10.4f} {st:<10.4f} "
              f"{m_prev:<10.4f} {m_next:<10.4f}")

    print(f'\n[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
