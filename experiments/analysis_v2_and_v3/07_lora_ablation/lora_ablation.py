"""
07 — LoRA ablation: MCFM blending vs LoRA self-consistency refinement.

Compares rendered frame quality (PSNR, temporal flicker, LPIPS) for:
  A) MCFM (beta=0, no LoRA)  — results_mcfm_{mode}_seed6_fixednoise/beta0p0/
  B) LoRA self-cons           — results_mcfm_{mode}_selfcons_lora_seed6/renders_e{N}/rendered/
     (finds the latest epoch checkpoint render directory automatically)

If LoRA renders are not yet available (training still running), the script
prints a warning and reports MCFM-only numbers so it can already be run.

Also creates:
  1. Side-by-side keyframe comparison: MCFM | LoRA | GT  (one figure per mode)
  2. Bar chart: PSNR and flicker for MCFM vs LoRA across all 4 modes
  3. Comparison video: MCFM | LoRA | GT  (if LoRA renders exist)

NO GPU required.

Output: 07_lora_ablation/results/

Usage:
  python lora_ablation.py
  python lora_ablation.py --mode v2_C
  python lora_ablation.py --skip_video
"""

import sys, argparse, subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent.parent
_ENH      = _ROOT / 'experiments' / 'enhancement'
VIDEO_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                 '/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
N_FRAMES   = 150
RENDER_RES = 518
THUMB      = 240
MODES      = ['v2_C', 'v2_D', 'v3_C', 'v3_D']
KEYFRAMES  = [1, 25, 50, 75, 100, 125, 150]
COLORS = {
    'v2_C': '#61afef', 'v2_D': '#56b6c2',
    'v3_C': '#e06c75', 'v3_D': '#d19a66',
}


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def mcfm_dir(mode: str) -> Path:
    return _ENH / f'results_mcfm_{mode}_seed6_fixednoise' / 'beta0p0'


def find_lora_renders(mode: str) -> Path | None:
    """Return path to rendered/ subdir of the latest epoch checkpoint, or None."""
    base = _ENH / f'results_mcfm_{mode}_selfcons_lora_seed6'
    if not base.exists():
        return None
    candidates = sorted(base.glob('renders_e*/rendered'))
    if not candidates:
        return None
    # latest epoch = last in sorted order (lora_e001, lora_e002 ... lora_e050)
    return candidates[-1]


def load_frames_np(frames_dir: Path) -> np.ndarray | None:
    """Load all 150 PNGs → (N, H, W, 3) float32 [0,1], or None if any missing."""
    frames = []
    for i in range(1, N_FRAMES + 1):
        p = frames_dir / f'frame_{i:04d}.png'
        if not p.exists():
            return None
        img = Image.open(p).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
        frames.append(np.array(img).astype(np.float32) / 255.0)
    return np.stack(frames)


def psnr_mean(rendered: np.ndarray, gt: np.ndarray) -> tuple:
    mse = ((rendered - gt) ** 2).mean(axis=(1, 2, 3))
    p   = 10 * np.log10(1.0 / (mse + 1e-8))
    return float(p.mean()), float(p.std())


def temporal_flicker(frames: np.ndarray) -> float:
    return float(np.abs(frames[1:] - frames[:-1]).mean())


def compute_lpips(rendered: np.ndarray, gt: np.ndarray) -> float:
    try:
        import torch, lpips
        loss_fn = lpips.LPIPS(net='alex'); loss_fn.eval()
        scores = []
        for r, g in zip(rendered, gt):
            tr = torch.from_numpy(r).permute(2,0,1).unsqueeze(0).float() * 2 - 1
            tg = torch.from_numpy(g).permute(2,0,1).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():
                scores.append(loss_fn(tr, tg).item())
        return float(np.mean(scores))
    except Exception:
        return float('nan')


def make_video(frames_dir: Path, out_path: Path, fps: int = 10):
    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        return
    fl = out_path.parent / '_fflist.txt'
    with open(fl, 'w') as f:
        for p in frames:
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0/fps:.6f}\n")
    r = subprocess.run(
        ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(fl), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path)],
        capture_output=True, text=True)
    fl.unlink(missing_ok=True)
    if r.returncode != 0:
        print(f'  [VIDEO] ffmpeg error: {r.stderr[-300:]}')
    else:
        print(f'  [VIDEO] saved → {out_path}')


def analyze_mode(mode: str, gt_frames: np.ndarray, out_dir: Path,
                 skip_video: bool, fps: int) -> dict:
    result = {'mode': mode}

    # ── MCFM ─────────────────────────────────────────────────────────────────
    md = mcfm_dir(mode)
    if not md.exists():
        print(f'  [{mode}] MCFM renders not found: {md}')
        return {}

    mcfm_frames = load_frames_np(md)
    if mcfm_frames is None:
        print(f'  [{mode}] MCFM renders incomplete (< 150 frames)')
        return {}

    m_psnr, m_psnr_std = psnr_mean(mcfm_frames, gt_frames)
    m_flicker           = temporal_flicker(mcfm_frames)
    m_lpips             = compute_lpips(mcfm_frames, gt_frames)
    result.update({'mcfm_psnr': m_psnr, 'mcfm_psnr_std': m_psnr_std,
                   'mcfm_flicker': m_flicker, 'mcfm_lpips': m_lpips})
    print(f'  [{mode}] MCFM:  PSNR={m_psnr:.2f}±{m_psnr_std:.2f}  '
          f'flicker={m_flicker:.5f}  LPIPS={m_lpips:.4f}')

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_dir = find_lora_renders(mode)
    if lora_dir is None:
        print(f'  [{mode}] LoRA renders not yet available — skipping LoRA metrics')
        result.update({'lora_psnr': float('nan'), 'lora_psnr_std': float('nan'),
                       'lora_flicker': float('nan'), 'lora_lpips': float('nan'),
                       'lora_epoch': -1})
        lora_frames = None
    else:
        epoch_str = lora_dir.parent.name  # e.g. 'renders_e050'
        epoch_num = int(epoch_str.replace('renders_e', ''))
        lora_frames = load_frames_np(lora_dir)

        if lora_frames is None:
            print(f'  [{mode}] LoRA renders incomplete in {lora_dir}')
            result.update({'lora_psnr': float('nan'), 'lora_psnr_std': float('nan'),
                           'lora_flicker': float('nan'), 'lora_lpips': float('nan'),
                           'lora_epoch': epoch_num})
            lora_frames = None
        else:
            l_psnr, l_psnr_std = psnr_mean(lora_frames, gt_frames)
            l_flicker           = temporal_flicker(lora_frames)
            l_lpips             = compute_lpips(lora_frames, gt_frames)
            result.update({'lora_psnr': l_psnr, 'lora_psnr_std': l_psnr_std,
                           'lora_flicker': l_flicker, 'lora_lpips': l_lpips,
                           'lora_epoch': epoch_num})
            print(f'  [{mode}] LoRA (e{epoch_num:03d}):  PSNR={l_psnr:.2f}±{l_psnr_std:.2f}  '
                  f'flicker={l_flicker:.5f}  LPIPS={l_lpips:.4f}')

    # ── Keyframe comparison figure ────────────────────────────────────────────
    rows = [('GT',   np.stack([np.array(
                 Image.open(VIDEO_DIR / f'frame_{kf:04d}.png').convert('RGB')
                      .resize((THUMB, THUMB), Image.LANCZOS))
                 for kf in KEYFRAMES]))]
    rows.append(('MCFM', np.stack([
        np.array(Image.open(md / f'frame_{kf:04d}.png').convert('RGB')
                      .resize((THUMB, THUMB), Image.LANCZOS))
        for kf in KEYFRAMES])))
    if lora_frames is not None:
        rows.append(('LoRA', np.stack([
            np.array(Image.open(lora_dir / f'frame_{kf:04d}.png').convert('RGB')
                          .resize((THUMB, THUMB), Image.LANCZOS))
            for kf in KEYFRAMES])))

    n_r = len(rows); n_c = len(KEYFRAMES)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.1 * n_c, 2.3 * n_r))
    if n_r == 1: axes = axes[np.newaxis, :]
    if n_c == 1: axes = axes[:, np.newaxis]
    fig.suptitle(f'Ablation: MCFM vs LoRA — {mode}', fontsize=11, y=1.01)

    for ri, (label, imgs) in enumerate(rows):
        for ci, (kf, img) in enumerate(zip(KEYFRAMES, imgs)):
            axes[ri, ci].imshow(img)
            axes[ri, ci].axis('off')
            if ri == 0:
                axes[ri, ci].set_title(f'f{kf}', fontsize=8)
        axes[ri, 0].set_ylabel(label, fontsize=9, rotation=0,
                                labelpad=40, va='center', ha='right')

    plt.tight_layout()
    fig.savefig(out_dir / f'ablation_keyframes_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [{mode}] saved ablation_keyframes_{mode}.png')

    # ── Comparison video (MCFM | LoRA | GT) ──────────────────────────────────
    if not skip_video and lora_frames is not None:
        vdir = out_dir / f'video_{mode}'
        vdir.mkdir(exist_ok=True)
        cell = 300
        label_h = 26
        total_w = cell * 3; total_h = cell + label_h
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        except Exception:
            font = ImageFont.load_default()

        print(f'  [{mode}] Building comparison video frames...')
        for fi in range(1, N_FRAMES + 1):
            fp = vdir / f'frame_{fi:04d}.png'
            if fp.exists():
                continue
            canvas = Image.new('RGB', (total_w, total_h), (28, 28, 28))
            draw   = ImageDraw.Draw(canvas)
            srcs   = [
                (mcfm_dir(mode) / f'frame_{fi:04d}.png', 'MCFM (beta=0)'),
                (lora_dir       / f'frame_{fi:04d}.png', f'LoRA e{result.get("lora_epoch",0):03d}'),
                (VIDEO_DIR      / f'frame_{fi:04d}.png', 'GT video'),
            ]
            for ci, (src, lbl) in enumerate(srcs):
                img = (Image.open(src).convert('RGB').resize((cell, cell), Image.LANCZOS)
                       if src.exists() else Image.new('RGB', (cell, cell), (50,50,50)))
                canvas.paste(img, (ci * cell, label_h))
                draw.rectangle([ci * cell, 0, (ci+1)*cell - 1, label_h - 1], fill=(40, 40, 55))
                draw.text((ci * cell + 6, 6), lbl, fill=(200,200,200), font=font)
            draw.text((total_w - 65, total_h - 18), f'f{fi:03d}', fill=(160,160,160), font=font)
            canvas.save(fp)

        make_video(vdir, out_dir / f'ablation_video_{mode}.mp4', fps=fps)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None,
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--skip_video', action='store_true')
    parser.add_argument('--fps', type=int, default=10)
    args  = parser.parse_args()
    modes = [args.mode] if args.mode else MODES

    out_dir = _HERE / 'results' / 'lora_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'lora_ablation.log')
    sys.stderr = sys.stdout

    print('[LORA ABLATION] Loading GT frames...')
    gt_frames = np.stack([
        np.array(Image.open(VIDEO_DIR / f'frame_{i:04d}.png').convert('RGB')
                      .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)).astype(np.float32) / 255.0
        for i in range(1, N_FRAMES + 1)
    ])
    print(f'  GT frames: {gt_frames.shape}  flicker={temporal_flicker(gt_frames):.5f}')

    all_r = []
    for mode in modes:
        print(f'\n[{mode}]')
        r = analyze_mode(mode, gt_frames, out_dir,
                         skip_video=args.skip_video, fps=args.fps)
        if r:
            all_r.append(r)

    if not all_r:
        print('[DONE] No results.')
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n[SUMMARY TABLE]')
    print(f"{'Mode':<8} {'MCFM_PSNR':<14} {'LoRA_PSNR':<14} "
          f"{'MCFM_flkr':<12} {'LoRA_flkr':<12}")
    print('-' * 65)
    for r in all_r:
        lp = (f"{r['lora_psnr']:.2f}±{r['lora_psnr_std']:.2f}"
              if not np.isnan(r['lora_psnr']) else 'N/A (pending)')
        ll = (f"{r['lora_flicker']:.5f}"
              if not np.isnan(r['lora_flicker']) else 'N/A')
        print(f"  {r['mode']:<8} {r['mcfm_psnr']:.2f}±{r['mcfm_psnr_std']:.2f}      "
              f"{lp:<14} {r['mcfm_flicker']:.5f}      {ll}")

    # Save CSV
    with open(out_dir / 'ablation_summary.csv', 'w') as f:
        f.write('mode,mcfm_psnr,mcfm_psnr_std,mcfm_flicker,mcfm_lpips,'
                'lora_psnr,lora_psnr_std,lora_flicker,lora_lpips,lora_epoch\n')
        for r in all_r:
            f.write(f"{r['mode']},{r['mcfm_psnr']:.4f},{r['mcfm_psnr_std']:.4f},"
                    f"{r['mcfm_flicker']:.6f},{r['mcfm_lpips']:.6f},"
                    f"{r['lora_psnr']:.4f},{r['lora_psnr_std']:.4f},"
                    f"{r['lora_flicker']:.6f},{r['lora_lpips']:.6f},"
                    f"{r['lora_epoch']}\n")

    # ── Bar chart: PSNR and flicker ───────────────────────────────────────────
    ready = [r for r in all_r if not np.isnan(r['lora_psnr'])]
    if ready:
        labels = [r['mode'] for r in ready]
        x      = np.arange(len(labels))
        w      = 0.38
        cols   = [COLORS[m] for m in labels]
        lora_c = ['#c678dd'] * len(labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Ablation: MCFM vs LoRA self-consistency', fontsize=12)

        # PSNR
        b1 = ax1.bar(x - w/2, [r['mcfm_psnr'] for r in ready], w,
                     label='MCFM (beta=0)', color=cols, alpha=0.85)
        b2 = ax1.bar(x + w/2, [r['lora_psnr'] for r in ready], w,
                     label='LoRA self-cons', color=lora_c, alpha=0.85)
        for bar, r in zip(b1, ready):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{r['mcfm_psnr']:.1f}", ha='center', va='bottom', fontsize=7)
        for bar, r in zip(b2, ready):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{r['lora_psnr']:.1f}", ha='center', va='bottom', fontsize=7)
        ax1.set_xticks(x); ax1.set_xticklabels(labels)
        ax1.set_ylabel('PSNR (dB)  ↑'); ax1.set_title('PSNR vs GT')
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')

        # Flicker
        b3 = ax2.bar(x - w/2, [r['mcfm_flicker'] for r in ready], w,
                     label='MCFM (beta=0)', color=cols, alpha=0.85)
        b4 = ax2.bar(x + w/2, [r['lora_flicker'] for r in ready], w,
                     label='LoRA self-cons', color=lora_c, alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(labels)
        ax2.set_ylabel('Temporal flicker  (mean |Δframe|)  ↓')
        ax2.set_title('Temporal flicker')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(out_dir / 'ablation_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('[PLOT] ablation_bar.png')
    else:
        print('[PLOT] bar chart skipped — LoRA renders not yet available for any mode')

    print(f'\n[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
