"""
Step 0 — Instrument (no training, no GPU).

For all 4 modes (v2_C, v2_D, v3_C, v3_D), loads every saved checkpoint and computes
per-epoch, per-block:
  - alpha
  - ||A|| (lora_q.A and lora_kv.A Frobenius norm)
  - ||B|| (lora_q.B and lora_kv.B Frobenius norm)
  - ||B@A|| (effective delta matrix norm)
  - alpha * mean_block(||B@A||)  ← the true effective delta magnitude
  - g_task vs g_reg on alpha (separated from loss_history alpha_grad_mean)

Plots saved to lora_diagnostic/:
  step0_effective_delta.png   — alpha * ||B@A|| over epochs, all 4 modes
  step0_AB_norms.png          — ||A||, ||B||, ||B@A|| per mode
  step0_grad_split.png        — g_task vs g_reg on alpha per mode
  step0_summary.log           — all numbers
"""

import json, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE    = Path(__file__).resolve().parent
ENH     = HERE.parent
ALPHA_REG = 0.01   # same as step07c
MODES   = ['v2_C', 'v2_D', 'v3_C', 'v3_D']
N_BLOCKS = 24

LOG_PATH = HERE / 'step0_summary.log'
LOG_PATH.unlink(missing_ok=True)

def log(msg=''):
    ts = f'[{time.strftime("%H:%M:%S")}] '
    full = ts + msg
    print(full, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(full + '\n')


def block_norms(lora_state, blk_idx):
    """Return (qA_norm, qB_norm, qBA_norm, kvA_norm, kvB_norm, kvBA_norm) for one block."""
    qA  = lora_state[f'{blk_idx}.lora_q.A'].float()
    qB  = lora_state[f'{blk_idx}.lora_q.B'].float()
    kvA = lora_state[f'{blk_idx}.lora_kv.A'].float()
    kvB = lora_state[f'{blk_idx}.lora_kv.B'].float()
    qBA  = (qB  @ qA).norm().item()
    kvBA = (kvB @ kvA).norm().item()
    return (qA.norm().item(), qB.norm().item(), qBA,
            kvA.norm().item(), kvB.norm().item(), kvBA)


def load_mode(mode):
    results_dir = ENH / f'results_mcfm_{mode}_realgt_lora_seed6'
    ckpt_dir    = results_dir / 'lora_ckpts'
    history_path = results_dir / 'loss_history.json'

    with open(history_path) as f:
        history = json.load(f)

    ckpt_files = sorted(ckpt_dir.glob('lora_e*.pt'))
    log(f'  {mode}: {len(ckpt_files)} checkpoints, {len(history)} history entries')

    epochs, alphas = [], []
    mean_qBA, mean_kvBA, mean_BA = [], [], []
    eff_deltas = []         # alpha * mean ||B@A||
    g_task_list, g_reg_list = [], []

    # per-block trajectories: shape (n_epochs, n_blocks)
    qA_mat, qB_mat, qBA_mat = [], [], []
    kvA_mat, kvB_mat, kvBA_mat = [], [], []

    for ckpt_path in ckpt_files:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        epoch = int(ckpt['epoch'])
        alpha = float(ckpt['alpha'].item())

        blk_qA, blk_qB, blk_qBA = [], [], []
        blk_kvA, blk_kvB, blk_kvBA = [], [], []
        for b in range(N_BLOCKS):
            qA_n, qB_n, qBA_n, kvA_n, kvB_n, kvBA_n = block_norms(ckpt['lora_state'], b)
            blk_qA.append(qA_n);   blk_qB.append(qB_n);   blk_qBA.append(qBA_n)
            blk_kvA.append(kvA_n); blk_kvB.append(kvB_n); blk_kvBA.append(kvBA_n)

        mBA  = float(np.mean(blk_qBA + blk_kvBA))   # mean over all blocks and both q/kv
        eff  = alpha * mBA

        # recover g_task and g_reg from history
        hist_entry = next((h for h in history if h['epoch'] == epoch), None)
        if hist_entry:
            ag_total = hist_entry['alpha_grad_mean']
            g_reg    = 2.0 * ALPHA_REG * alpha          # analytical: d/d_alpha (ALPHA_REG*alpha^2)
            g_task   = ag_total - g_reg
        else:
            ag_total = g_reg = g_task = float('nan')

        epochs.append(epoch)
        alphas.append(alpha)
        mean_qBA.append(float(np.mean(blk_qBA)))
        mean_kvBA.append(float(np.mean(blk_kvBA)))
        mean_BA.append(mBA)
        eff_deltas.append(eff)
        g_task_list.append(g_task)
        g_reg_list.append(g_reg)
        qA_mat.append(blk_qA);   qB_mat.append(blk_qB);   qBA_mat.append(blk_qBA)
        kvA_mat.append(blk_kvA); kvB_mat.append(blk_kvB); kvBA_mat.append(blk_kvBA)

        log(f'    e{epoch:03d}  alpha={alpha:.4f}  mean||BA||={mBA:.4f}  '
            f'eff_delta={eff:.4f}  g_task={g_task:.4e}  g_reg={g_reg:.4e}')

    return {
        'mode'      : mode,
        'epochs'    : epochs,
        'alphas'    : alphas,
        'mean_qBA'  : mean_qBA,
        'mean_kvBA' : mean_kvBA,
        'mean_BA'   : mean_BA,
        'eff_deltas': eff_deltas,
        'g_task'    : g_task_list,
        'g_reg'     : g_reg_list,
        'qA_mat'    : qA_mat,    # (n_epochs, n_blocks)
        'qB_mat'    : qB_mat,
        'qBA_mat'   : qBA_mat,
        'kvA_mat'   : kvA_mat,
        'kvB_mat'   : kvB_mat,
        'kvBA_mat'  : kvBA_mat,
    }


def main():
    log('=== Step 0 — Instrument (no training) ===')
    log(f'  modes: {MODES}')
    log(f'  ALPHA_REG={ALPHA_REG}  (used to separate g_task from g_reg)')

    all_data = {}
    for mode in MODES:
        log(f'\n[{mode}]')
        all_data[mode] = load_mode(mode)

    # ── Plot 1: effective delta (alpha * mean||B@A||) over epochs ─────────────
    colors = {'v2_C': '#e06c75', 'v2_D': '#61afef', 'v3_C': '#98c379', 'v3_D': '#e5c07b'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Step 0 — Effective delta magnitude over training', fontsize=13)

    for mode, d in all_data.items():
        axes[0].plot(d['epochs'], d['eff_deltas'], 'o-', lw=2, ms=5,
                     color=colors[mode], label=mode)
        axes[1].plot(d['epochs'], d['alphas'], 'o-', lw=2, ms=5,
                     color=colors[mode], label=mode)
        axes[2].plot(d['epochs'], d['mean_BA'], 'o-', lw=2, ms=5,
                     color=colors[mode], label=mode)

    axes[0].set_title('alpha × mean||B@A||  (effective delta)')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('alpha'); axes[1].set_ylim(0, 0.6)
    axes[1].axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='init')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_title('mean ||B@A|| per block')
    axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = HERE / 'step0_effective_delta.png'
    fig.savefig(out1, dpi=130, bbox_inches='tight'); plt.close()
    log(f'\nsaved: {out1}')

    # ── Plot 2: ||A||, ||B||, ||B@A|| trajectories per mode ──────────────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 12))
    fig.suptitle('Step 0 — ||A||, ||B||, ||B@A|| per block mean over epochs', fontsize=13)

    for col, mode in enumerate(MODES):
        d = all_data[mode]
        ep = d['epochs']
        mean_qA  = [float(np.mean(r)) for r in d['qA_mat']]
        mean_qB  = [float(np.mean(r)) for r in d['qB_mat']]
        mean_qBA = d['mean_qBA']
        mean_kvA  = [float(np.mean(r)) for r in d['kvA_mat']]
        mean_kvB  = [float(np.mean(r)) for r in d['kvB_mat']]
        mean_kvBA = d['mean_kvBA']

        axes[0, col].plot(ep, mean_qA,  'o-', color='#e06c75', lw=2, ms=4, label='||qA||')
        axes[0, col].plot(ep, mean_qB,  's-', color='#61afef', lw=2, ms=4, label='||qB||')
        axes[0, col].plot(ep, mean_qBA, '^-', color='#98c379', lw=2, ms=4, label='||qB@qA||')
        axes[0, col].set_title(f'{mode} — lora_q'); axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(ep, mean_kvA,  'o-', color='#e06c75', lw=2, ms=4, label='||kvA||')
        axes[1, col].plot(ep, mean_kvB,  's-', color='#61afef', lw=2, ms=4, label='||kvB||')
        axes[1, col].plot(ep, mean_kvBA, '^-', color='#98c379', lw=2, ms=4, label='||kvB@kvA||')
        axes[1, col].set_title(f'{mode} — lora_kv'); axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].plot(ep, d['eff_deltas'], 'o-', color='#c678dd', lw=2, ms=4,
                          label='alpha×mean||BA||')
        axes[2, col].set_title(f'{mode} — effective delta'); axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = HERE / 'step0_AB_norms.png'
    fig.savefig(out2, dpi=130, bbox_inches='tight'); plt.close()
    log(f'saved: {out2}')

    # ── Plot 3: g_task vs g_reg on alpha ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Step 0 — g_task vs g_reg on alpha (separated)', fontsize=13)

    for col, mode in enumerate(MODES):
        d = all_data[mode]
        ep = d['epochs']
        axes[col].plot(ep, d['g_task'], 'o-', color='#e06c75', lw=2, ms=5, label='g_task')
        axes[col].plot(ep, d['g_reg'],  's-', color='#61afef', lw=2, ms=5, label='g_reg')
        axes[col].axhline(0, color='black', lw=0.8)
        axes[col].set_title(f'{mode}')
        axes[col].set_xlabel('Epoch')
        axes[col].legend(fontsize=9); axes[col].grid(True, alpha=0.3)
        # annotation
        if all(g > 0 for g in d['g_task'] if not np.isnan(g)):
            axes[col].set_title(f'{mode}\ng_task ALWAYS POSITIVE → task never wants more LoRA')
        elif all(g < 0 for g in d['g_task'] if not np.isnan(g)):
            axes[col].set_title(f'{mode}\ng_task ALWAYS NEGATIVE → task wants more LoRA ✓')

    plt.tight_layout()
    out3 = HERE / 'step0_grad_split.png'
    fig.savefig(out3, dpi=130, bbox_inches='tight'); plt.close()
    log(f'saved: {out3}')

    # ── Summary table ─────────────────────────────────────────────────────────
    log('\n=== SUMMARY TABLE ===')
    log(f'{"mode":<6}  {"epochs":>6}  {"alpha_final":>11}  {"mean_BA_final":>13}  '
        f'{"eff_delta_final":>15}  {"g_task_sign"}')
    for mode, d in all_data.items():
        gt_signs = ['+' if g > 0 else '-' for g in d['g_task'] if not np.isnan(g)]
        sign_str = 'always+' if all(s == '+' for s in gt_signs) else \
                   'always-' if all(s == '-' for s in gt_signs) else 'mixed'
        log(f'{mode:<6}  {len(d["epochs"]):>6}  {d["alphas"][-1]:>11.4f}  '
            f'{d["mean_BA"][-1]:>13.4f}  {d["eff_deltas"][-1]:>15.4f}  {sign_str}')
    log('\nDONE.')


if __name__ == '__main__':
    main()
