"""
Evaluate all seed renders for a given frame using:
  1. LPIPS  — perceptual similarity (lower = closer to GT)
  2. HSV hue histogram correlation (higher = closer to GT hue distribution)

Masks out white background before comparing so only the teapot region counts.

Usage:
  python eval.py --frame 77
"""

import sys, argparse
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

sys.stdout = _Tee(RESULTS_DIR / 'eval.log')
sys.stderr = sys.stdout

import numpy as np
from PIL import Image
GT_FRAMES   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs'
                   '/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')


def fg_mask(img_np, white_thresh=240):
    """Return boolean mask of non-white (foreground) pixels."""
    return np.all(img_np < white_thresh, axis=-1)


def hsv_hue_sim(gt_np, pred_np):
    """Histogram correlation on H channel, foreground pixels only."""
    from PIL import Image as _I
    mask = fg_mask(gt_np) & fg_mask(pred_np)
    if mask.sum() == 0:
        return 0.0
    gt_h   = np.array(_I.fromarray(gt_np).convert('HSV'))[:, :, 0][mask].astype(np.float32)
    pred_h = np.array(_I.fromarray(pred_np).convert('HSV'))[:, :, 0][mask].astype(np.float32)
    gt_hist,   _ = np.histogram(gt_h,   bins=64, range=(0, 256), density=True)
    pred_hist, _ = np.histogram(pred_h, bins=64, range=(0, 256), density=True)
    # Correlation coefficient
    gt_c   = gt_hist   - gt_hist.mean()
    pred_c = pred_hist - pred_hist.mean()
    denom  = (np.linalg.norm(gt_c) * np.linalg.norm(pred_c) + 1e-8)
    return float(np.dot(gt_c, pred_c) / denom)


def lpips_score(gt_np, pred_np, model):
    import torch
    def to_t(x):
        t = torch.from_numpy(x).float() / 127.5 - 1.0  # [-1,1]
        return t.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(model(to_t(gt_np), to_t(pred_np)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=77)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    frame_i = args.frame
    fname   = f'frame_{frame_i:04d}.png'
    gt_np   = np.array(Image.open(GT_FRAMES / fname).convert('RGB').resize((518, 518)))

    # ── Load LPIPS ────────────────────────────────────────────────────────────
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg')
        loss_fn.eval()
        use_lpips = True
        print('LPIPS: using VGG backend')
    except ImportError:
        try:
            from skimage.metrics import structural_similarity as ssim
            use_lpips = False
            print('LPIPS not found, falling back to 1-SSIM')
        except ImportError:
            use_lpips = None
            print('WARNING: neither lpips nor skimage found — skipping perceptual metric')

    # ── Find all seed renders for this frame ─────────────────────────────────
    renders = sorted(p for p in RESULTS_DIR.glob(f'seed[0-9]*_f{frame_i:04d}.png'))
    if not renders:
        print(f'No seed renders found in {RESULTS_DIR}')
        return

    print(f'\nEvaluating {len(renders)} seeds for frame {frame_i}...\n')
    print(f'{"seed":>6}  {"LPIPS/SSIM":>12}  {"HueSim":>8}  {"combined":>10}')
    print('-' * 44)

    rows = []
    for path in renders:
        seed = int(path.stem.split('_')[0].replace('seed', ''))
        pred_np = np.array(Image.open(path).convert('RGB'))

        # perceptual
        if use_lpips is True:
            perc = lpips_score(gt_np, pred_np, loss_fn)
            perc_label = 'lpips'
        elif use_lpips is False:
            from skimage.metrics import structural_similarity as ssim
            perc = 1.0 - ssim(gt_np, pred_np, channel_axis=2, data_range=255)
            perc_label = '1-ssim'
        else:
            perc = float('nan')
            perc_label = 'n/a'

        hue = hsv_hue_sim(gt_np, pred_np)

        # combined rank score: lower is better
        # normalise lpips [0..1] lower=better, hue_sim [-1..1] higher=better
        # combined = lpips - hue_sim  (lower=better)
        combined = (perc if not np.isnan(perc) else 0.5) - hue
        rows.append((seed, perc, hue, combined))

    rows.sort(key=lambda r: r[3])  # sort by combined (lower=better)

    for seed, perc, hue, comb in rows:
        print(f'{seed:>6}  {perc:>12.4f}  {hue:>8.4f}  {comb:>10.4f}')

    print(f'\n=== TOP {args.top} seeds (best combined score) ===')
    for rank, (seed, perc, hue, comb) in enumerate(rows[:args.top], 1):
        print(f'  #{rank}  seed={seed:3d}  {perc_label}={perc:.4f}  hue_sim={hue:.4f}  combined={comb:.4f}')

    # ── Save top-N comparison grid ────────────────────────────────────────────
    top_seeds = [r[0] for r in rows[:args.top]]
    gt_img    = Image.fromarray(gt_np)
    all_imgs  = [gt_img] + [Image.open(RESULTS_DIR / f'seed{s:03d}_f{frame_i:04d}.png').convert('RGB')
                             for s in top_seeds]
    all_labels = [f'GT f{frame_i}'] + [f'#{i+1} seed={s}' for i, s in enumerate(top_seeds)]

    W, H    = 518, 518
    LABEL_H = 28
    COLS    = 5
    ROWS    = (len(all_imgs) + COLS - 1) // COLS
    grid    = Image.new('RGB', (W * COLS, (H + LABEL_H) * ROWS), (30, 30, 30))
    from PIL import ImageDraw
    draw    = ImageDraw.Draw(grid)

    for idx, (img, label) in enumerate(zip(all_imgs, all_labels)):
        col = idx % COLS
        row = idx // COLS
        x   = col * W
        y   = row * (H + LABEL_H)
        grid.paste(img.resize((W, H)), (x, y + LABEL_H))
        draw.rectangle([x, y, x + W - 1, y + LABEL_H - 1], fill=(30, 30, 30))
        draw.text((x + W // 2, y + LABEL_H // 2), label, fill='white', anchor='mm')

    out = RESULTS_DIR / f'top{args.top}_grid_f{frame_i:04d}.png'
    grid.save(out)
    print(f'\nTop-{args.top} grid saved: {out}')


if __name__ == '__main__':
    main()
