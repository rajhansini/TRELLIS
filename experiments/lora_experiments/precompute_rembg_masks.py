"""
precompute_rembg_masks.py
-------------------------
Cache the u2net foreground masks for the 150 GT frames, once.

WHY
  The rungs' training loss uses

      gt_mask = (gt < 0.99).any(dim=0)

  which is true wherever ANY channel is below 0.99. The video's background is
  off-white (~0.98), so it passes: the mask covers 99.7% of the frame instead of
  the 11.5% that is teapot. Since

      mse = sum((render-gt)^2 * m) / (sum(m) * 3)

  that inflates the denominator ~8.7x and silently shrinks MSE by the same
  factor. Measured on rung15's own logs, the nominal 10:1 MSE:LPIPS weighting
  ran at 1.8:1.

WHY rembg RATHER THAN A BRIGHTNESS THRESHOLD
  A tight brightness rule (min(R,G,B) < 0.95) matches u2net at IoU 0.979 on this
  video and is far cheaper. But it only works BECAUSE this background is bright
  and uniform; against a dark backdrop it collapses. u2net is what TRELLIS
  itself uses to segment its conditioning images
  (trellis_image_to_3d.py:100-105, alpha > 0.8*255), so taking the mask from the
  same place keeps the loss definition transferable to any video.

WHY A SEPARATE SCRIPT
  u2net is CPU-only here: ~21 s/frame, ~53 min for 150. Doing it at the start of
  every training run would pay that repeatedly. It is a property of the DATA, not
  of any rung, so it is computed once and cached.

WHAT IS WRITTEN
  gt_masks_rembg.npz
      masks   [150, 518, 518] bool   True = foreground
      frames  [150] int32
      meta    json: threshold, resolution, session, per-frame pixel counts

  The resize to 518 with LANCZOS happens BEFORE rembg, matching load_gt() in the
  rungs exactly, so the mask lines up pixel-for-pixel with the GT the loss sees.

Usage:
  python experiments/lora_experiments/precompute_rembg_masks.py
"""

import argparse, json, time
from pathlib import Path

import numpy as np
from PIL import Image
import rembg

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--frames-dir', default='/net/projects/ranalab/rajhansini/'
                'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/'
                'teapot_lava_kling_premium_front/all_frames_150')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--res', type=int, default=518, help='RENDER_RES; must match the rungs')
ap.add_argument('--alpha-thresh', type=float, default=0.8,
                help='trellis_image_to_3d.py:105 uses alpha > 0.8*255')
ap.add_argument('--session', default='u2net')
ap.add_argument('--out', default=str(_HERE / 'gt_masks_rembg.npz'))
args = ap.parse_args()

GT = Path(args.frames_dir)
OUT = Path(args.out)


def main():
    t0 = time.time()
    print(f'[REMBG] session={args.session}  res={args.res}  '
          f'alpha>{args.alpha_thresh}*255', flush=True)
    sess = rembg.new_session(args.session)

    masks, counts = [], []
    for i in range(1, args.n_frames + 1):
        p = GT / f'frame_{i:04d}.png'
        assert p.exists(), f'missing frame: {p}'
        # resize FIRST, exactly as load_gt() does, so the mask is pixel-aligned
        im = Image.open(p).convert('RGB').resize((args.res, args.res), Image.LANCZOS)
        alpha = np.array(rembg.remove(im, session=sess))[:, :, 3]
        m = alpha > args.alpha_thresh * 255
        masks.append(m)
        counts.append(int(m.sum()))
        if i % 10 == 0 or i == args.n_frames:
            el = time.time() - t0
            print(f'  {i:3d}/{args.n_frames}  {counts[-1]:6,} px '
                  f'({100*counts[-1]/(args.res**2):5.2f}%)  '
                  f'{el:5.0f}s  eta {el/i*(args.n_frames-i):5.0f}s', flush=True)

    M = np.stack(masks)
    frac = M.reshape(len(M), -1).mean(1)
    # A mask that is ~0 or ~1 means segmentation failed on that frame and would
    # silently corrupt the loss for it. Fail here instead.
    bad = [(i + 1, float(f)) for i, f in enumerate(frac) if f < 0.02 or f > 0.60]
    assert not bad, f'implausible mask fraction on frames: {bad[:10]}'

    meta = dict(n_frames=args.n_frames, res=args.res,
                alpha_thresh=args.alpha_thresh, session=args.session,
                frames_dir=str(GT), counts=counts,
                frac_mean=float(frac.mean()), frac_std=float(frac.std()),
                frac_min=float(frac.min()), frac_max=float(frac.max()))
    np.savez_compressed(OUT, masks=M,
                        frames=np.arange(1, args.n_frames + 1, dtype=np.int32),
                        meta=json.dumps(meta))
    print(f'\n[DONE] {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)  {(time.time()-t0)/60:.1f} min')
    print(f'  foreground {100*frac.mean():.2f}% +- {100*frac.std():.2f}%  '
          f'(min {100*frac.min():.2f}%  max {100*frac.max():.2f}%)')
    print(f'  the leaky (gt<0.99).any() mask this replaces covered 99.7%')


if __name__ == '__main__':
    main()
