"""
Phase A: Supervision Sanity Check

Takes existing GT frame and TRELLIS render (no re-rendering needed).
  1. rembg  -> GT object mask
  2. threshold -> render mask (white background)
  3. intersection mask
  4. MSE full vs MSE masked
  5. Save 8-panel visualization

Usage:
  python phase_a.py --frame 77
  python phase_a.py --frame 1 --frame 75 --frame 150
"""

import os, sys, argparse
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent

GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..'
                 / 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
RENDER_DIR    = _PIPE / 'results' / 'phase_b' / 'flickering'
OUT_DIR       = _PIPE / 'results' / 'phase_a'
RENDER_RES    = 518


def get_gt_mask(gt_rgb: np.ndarray) -> np.ndarray:
    from rembg import remove
    result = remove(Image.fromarray(gt_rgb))
    return np.array(result)[:, :, 3] > 128          # HxW bool


def get_render_mask(ren_rgb: np.ndarray) -> np.ndarray:
    # render has white background — anything not near-white is object
    gray = ren_rgb.mean(axis=2)
    return gray < 250                                # HxW bool


def make_panel(imgs: list, labels: list) -> np.ndarray:
    """Stack images horizontally with labels."""
    pad, label_h = 8, 22
    H, W = RENDER_RES, RENDER_RES
    n    = len(imgs)
    canvas = np.full((H + 2*pad + label_h, n*(W+pad)+pad, 3), 30, dtype=np.uint8)
    pil    = Image.fromarray(canvas)
    draw   = ImageDraw.Draw(pil)

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = i*(W+pad) + pad
        if img.ndim == 2:                           # grayscale mask
            rgb = np.stack([img.astype(np.uint8)*255]*3, axis=2)
        else:
            rgb = img
        pil.paste(Image.fromarray(rgb), (x, pad + label_h))
        draw.text((x + W//2 - len(label)*3, 5), label, fill=(220,220,220))

    return np.array(pil)


def run_frame(frame_idx: int):
    gt_path  = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    ren_path = RENDER_DIR    / f'frame_{frame_idx:04d}.png'

    if not ren_path.exists():
        print(f'  [SKIP] render not found: {ren_path}')
        return

    gt_rgb  = np.array(Image.open(gt_path ).convert('RGB').resize((RENDER_RES,RENDER_RES), Image.LANCZOS))
    ren_rgb = np.array(Image.open(ren_path).convert('RGB'))

    print(f'  Getting GT mask (rembg)...')
    gt_mask  = get_gt_mask(gt_rgb)
    ren_mask = get_render_mask(ren_rgb)
    inter    = gt_mask & ren_mask

    gt_f  = gt_rgb.astype(np.float32)  / 255.0
    ren_f = ren_rgb.astype(np.float32) / 255.0

    mse_full   = float(np.mean((gt_f - ren_f)**2))
    mse_masked = float(np.mean((gt_f - ren_f)[inter]**2)) if inter.any() else float('nan')
    overlap_pct = inter.mean() * 100

    print(f'  MSE full={mse_full:.5f}  MSE masked={mse_masked:.5f}  overlap={overlap_pct:.1f}%')

    # per-pixel MSE heatmap (hot colormap manually)
    mse_map = ((gt_f - ren_f)**2).mean(axis=2)
    mse_norm = (mse_map / (mse_map.max() + 1e-8) * 255).astype(np.uint8)
    from PIL import Image as _I
    import PIL.ImageOps
    heatmap = np.array(_I.fromarray(mse_norm).convert('RGB'))
    # simple hot: low=black, mid=red, high=yellow
    h = mse_norm.astype(np.float32) / 255.0
    hot = np.stack([
        np.clip(h * 2,       0, 1),
        np.clip(h * 2 - 1,   0, 1),
        np.zeros_like(h),
    ], axis=2)
    heatmap = (hot * 255).astype(np.uint8)

    gt_masked  = (gt_f  * inter[:,:,None] * 255).astype(np.uint8)
    ren_masked = (ren_f * inter[:,:,None] * 255).astype(np.uint8)

    panel = make_panel(
        [gt_rgb, ren_rgb, gt_mask, ren_mask, inter, gt_masked, ren_masked, heatmap],
        ['GT', 'Render', 'GT mask', 'Render mask', 'Intersection',
         'GT x inter', 'Render x inter', f'MSE map\nfull={mse_full:.4f}\nmask={mse_masked:.4f}']
    )

    out_path = OUT_DIR / f'phase_a_{frame_idx:04d}.png'
    Image.fromarray(panel).save(out_path)
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, action='append', default=None)
    args   = parser.parse_args()
    frames = args.frame if args.frame else [77]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in frames:
        print(f'\n--- Frame {f:04d} ---')
        run_frame(f)


if __name__ == '__main__':
    main()
