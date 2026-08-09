"""
04 — Quality metrics: FID, PSNR, temporal flicker, LPIPS.

Compares MCFM rendered frames (beta0p0) against GT video frames.

Metrics:
  1. PSNR         : per-frame, mean over 150 frames
  2. Temporal flicker : mean |frame_{t+1} - frame_t| — rendered AND gt (for reference)
  3. FID          : Frechet Inception Distance (rendered vs GT set, Inception V3 features)
  4. LPIPS        : if torchmetrics/lpips available

NO GPU required for PSNR/flicker. Inception model runs on CPU (slow but fine for 150 frames).

Output: results/fid_metrics/

Usage:
  python fid_metrics.py --mode v2_C
  python fid_metrics.py               # all 4 modes
"""

import sys, argparse, warnings
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
N_FRAMES   = 150
RENDER_RES = 518
INCEPTION_RES = 299


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def load_frames_np(frames_dir: Path, res: int = RENDER_RES) -> np.ndarray:
    """Load PNGs → (N, H, W, 3) float32 [0,1]."""
    frames = []
    for i in range(1, N_FRAMES + 1):
        p = frames_dir / f'frame_{i:04d}.png'
        img = Image.open(p).convert('RGB').resize((res, res), Image.LANCZOS)
        frames.append(np.array(img).astype(np.float32) / 255.0)
    return np.stack(frames)


def psnr(rendered: np.ndarray, gt: np.ndarray) -> tuple:
    mse_per  = ((rendered - gt) ** 2).mean(axis=(1, 2, 3))
    psnr_per = 10 * np.log10(1.0 / (mse_per + 1e-8))
    return float(psnr_per.mean()), float(psnr_per.std())


def temporal_flicker(frames: np.ndarray) -> float:
    return float(np.abs(frames[1:] - frames[:-1]).mean())


def compute_fid(rendered: np.ndarray, gt: np.ndarray) -> float:
    """
    FID using Inception V3 pool3 features (2048-dim).
    Both inputs: (N, H, W, 3) float32 [0,1], resized to 299x299.
    """
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms.functional as TF
        from scipy.linalg import sqrtm

        print('  [FID] Loading Inception V3...')
        inception = models.inception_v3(pretrained=False)
        # Load from local cache or download
        try:
            state = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                map_location='cpu')
            inception.load_state_dict(state)
        except Exception:
            print('  [FID] Could not load Inception weights — skipping FID')
            return float('nan')

        inception.eval()
        # Hook pool3 layer
        features = []
        def hook(m, i, o): features.append(o.detach().cpu().squeeze(-1).squeeze(-1).numpy())
        inception.avgpool.register_forward_hook(hook)

        def extract(imgs_np):
            feats = []
            for img in imgs_np:
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                t = TF.resize(t, [INCEPTION_RES, INCEPTION_RES])
                t = t * 2 - 1  # [-1, 1]
                features.clear()
                with torch.no_grad():
                    inception(t)
                feats.append(features[0])
            return np.concatenate(feats, axis=0)  # (N, 2048)

        print('  [FID] Extracting rendered features...')
        f_r = extract(rendered)
        print('  [FID] Extracting GT features...')
        f_g = extract(gt)

        mu_r, sig_r = f_r.mean(0), np.cov(f_r, rowvar=False)
        mu_g, sig_g = f_g.mean(0), np.cov(f_g, rowvar=False)

        diff   = mu_r - mu_g
        covmean, _ = sqrtm(sig_r @ sig_g, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))
        return fid

    except ImportError as e:
        print(f'  [FID] Import error: {e} — skipping')
        return float('nan')
    except Exception as e:
        print(f'  [FID] Error: {e} — skipping')
        return float('nan')


def compute_lpips(rendered: np.ndarray, gt: np.ndarray) -> float:
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.eval()
        scores = []
        for r, g in zip(rendered, gt):
            tr = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            tg = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():
                scores.append(loss_fn(tr, tg).item())
        return float(np.mean(scores))
    except ImportError:
        return float('nan')
    except Exception as e:
        print(f'  [LPIPS] Error: {e}')
        return float('nan')


def analyze_mode(mode: str, gt_frames: np.ndarray) -> dict:
    renders_dir = _ENH / f'results_mcfm_{mode}_seed6_fixednoise' / 'beta0p0'
    if not renders_dir.exists():
        print(f'  [{mode}] renders dir not found: {renders_dir}')
        return {}

    print(f'\n[{mode}] Loading renders from {renders_dir.name}/beta0p0...')
    rendered = load_frames_np(renders_dir)

    p_mean, p_std = psnr(rendered, gt_frames)
    flicker_r     = temporal_flicker(rendered)
    flicker_gt    = temporal_flicker(gt_frames)

    print(f'  PSNR   : {p_mean:.2f} ± {p_std:.2f} dB')
    print(f'  Flicker: rendered={flicker_r:.5f}  GT={flicker_gt:.5f}')

    print(f'  Computing FID...')
    fid_val = compute_fid(rendered, gt_frames)
    print(f'  FID    : {fid_val:.2f}')

    print(f'  Computing LPIPS...')
    lpips_val = compute_lpips(rendered, gt_frames)
    print(f'  LPIPS  : {lpips_val:.4f}' if not np.isnan(lpips_val) else '  LPIPS  : not available')

    return {
        'mode'       : mode,
        'psnr_mean'  : p_mean,
        'psnr_std'   : p_std,
        'flicker_r'  : flicker_r,
        'flicker_gt' : flicker_gt,
        'fid'        : fid_val,
        'lpips'      : lpips_val,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None,
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID computation (fast mode)')
    args  = parser.parse_args()
    modes = [args.mode] if args.mode else ['v2_C', 'v2_D', 'v3_C', 'v3_D']

    out_dir = _HERE / 'results' / 'fid_metrics'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'metrics.log')
    sys.stderr = sys.stdout

    print('[FID METRICS] Loading GT frames...')
    gt_frames = load_frames_np(VIDEO_DIR)
    print(f'  GT: {gt_frames.shape}  flicker={temporal_flicker(gt_frames):.5f}')

    all_results = []
    for mode in modes:
        r = analyze_mode(mode, gt_frames)
        if r:
            all_results.append(r)

    if not all_results:
        print('[DONE] No results computed.')
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n[SUMMARY TABLE]')
    header = f"{'Mode':<8} {'PSNR':<12} {'Flicker_R':<12} {'Flicker_GT':<12} {'FID':<10} {'LPIPS':<10}"
    print(header)
    print('-' * len(header))
    for r in all_results:
        fid_s   = f"{r['fid']:.2f}"  if not np.isnan(r['fid'])   else 'N/A'
        lpips_s = f"{r['lpips']:.4f}" if not np.isnan(r['lpips']) else 'N/A'
        print(f"  {r['mode']:<8} {r['psnr_mean']:.2f}±{r['psnr_std']:.2f}  "
              f"{r['flicker_r']:<12.5f} {r['flicker_gt']:<12.5f} "
              f"{fid_s:<10} {lpips_s:<10}")

    # Save CSV
    with open(out_dir / 'metrics_summary.csv', 'w') as f:
        f.write('mode,psnr_mean,psnr_std,flicker_rendered,flicker_gt,fid,lpips\n')
        for r in all_results:
            f.write(f"{r['mode']},{r['psnr_mean']:.4f},{r['psnr_std']:.4f},"
                    f"{r['flicker_r']:.6f},{r['flicker_gt']:.6f},"
                    f"{r['fid']:.4f},{r['lpips']:.6f}\n")
    print(f'\n[SAVE] metrics_summary.csv')

    # ── Bar plot ──────────────────────────────────────────────────────────────
    mode_labels = [r['mode'] for r in all_results]
    colors      = ['#61afef', '#56b6c2', '#e06c75', '#d19a66'][:len(all_results)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('MCFM rendering quality vs GT (beta=0)', fontsize=12)

    axes[0].bar(mode_labels, [r['psnr_mean'] for r in all_results],
                color=colors, alpha=0.85)
    axes[0].set_ylabel('PSNR (dB)'); axes[0].set_title('PSNR vs GT')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(mode_labels, [r['flicker_r'] for r in all_results],
                color=colors, alpha=0.85, label='Rendered')
    axes[1].axhline(all_results[0]['flicker_gt'], color='red', ls='--',
                    lw=1.5, label='GT flicker')
    axes[1].set_ylabel('Mean |Δframe|'); axes[1].set_title('Temporal Flicker')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    fid_vals = [r['fid'] for r in all_results if not np.isnan(r['fid'])]
    if fid_vals:
        axes[2].bar(mode_labels, [r['fid'] if not np.isnan(r['fid']) else 0
                                  for r in all_results],
                    color=colors, alpha=0.85)
        axes[2].set_ylabel('FID ↓'); axes[2].set_title('FID (lower is better)')
        axes[2].grid(True, alpha=0.3, axis='y')
    else:
        axes[2].set_title('FID (not available)')

    plt.tight_layout()
    fig.savefig(out_dir / 'metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[PLOT] metrics_bar.png')
    print(f'[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
