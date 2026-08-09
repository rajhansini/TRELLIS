"""
Experiment: different noise seeds.

ONE change vs baseline:
  - Baseline (C0): NOISE_SEED=42, frame 77 -> actual seed = 42+77 = 119
  - This:          try multiple base seeds for frame 77, everything else identical
                   (25 steps, voxel fixed from frame 75, no blending)

Goal: see if any seed produces a brighter/orange texture closer to GT.

Log: results/run.log

Usage:
  python run.py --frame 77 --seeds 0 1 2 3 4 5 10 42 100
"""

import sys, argparse, time, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

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

sys.stdout = _Tee(RESULTS_DIR / 'run.log')
sys.stderr = sys.stdout

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / 'dynamic_texture_trellis_pipeline'))

from _base import (
    DEVICE, INTRINSICS, EXTRINSICS, N_FRAMES,
    SLAT_MEAN, SLAT_STD,
    GT_FRAMES_DIR,
    encode_all_frames, denoise, normalize_slat, render_slat,
    make_renderer, load_pipeline
)
import trellis.modules.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=77)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 10, 42, 100])
    args = parser.parse_args()

    frame_i = args.frame
    seeds   = args.seeds

    print(f'=== seeds experiment ===')
    print(f'  change  : noise seed (baseline seed=42, actual=42+{frame_i}={42+frame_i})')
    print(f'  blending: none (pure current frame)')
    print(f'  steps   : 25  voxel: fixed from frame 75')
    print(f'  frame   : {frame_i}')
    print(f'  seeds   : {seeds}')
    print(f'  log     : {RESULTS_DIR}/run.log')

    pipeline, flow_model, coords, N_vox = load_pipeline()
    all_tokens = encode_all_frames()
    renderer   = make_renderer()
    flow_model.eval()

    tok_curr = all_tokens[frame_i]['tokens'].to(DEVICE)
    cond_gl  = tok_curr.unsqueeze(0)
    print(f'\n[WIRE] tok_curr shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}  std={tok_curr.float().std():.5f}')
    print(f'[WIRE] cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}  -> flow model input OK')
    print(f'[WIRE] coords shape={tuple(coords.shape)}  N_vox={N_vox}')

    rendered_frames = {}

    for seed in seeds:
        print(f'\n=== seed {seed} (actual noise seed = {seed}+{frame_i} = {seed+frame_i}) ===')
        torch.manual_seed(seed + frame_i)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )
        print(f'  [STEP5] noise_sp.feats shape={tuple(noise_sp.feats.shape)}')

        t0 = time.time()
        x0 = denoise(flow_model, noise_sp, cond_gl)
        print(f'  [STEP5] x0.feats  mean={x0.feats.float().mean():.5f}  std={x0.feats.float().std():.5f}')

        slat = normalize_slat(x0)
        del noise_sp, x0
        print(f'  [STEP6] slat.feats mean={slat.feats.float().mean():.5f}  std={slat.feats.float().std():.5f}')

        rendered = render_slat(pipeline, slat, renderer)
        del slat
        torch.cuda.empty_cache()

        print(f'  [STEP7] rendered shape={rendered.shape}  min={rendered.min()}  max={rendered.max()}')
        print(f'  [DONE]  elapsed={time.time()-t0:.1f}s')

        out_path = RESULTS_DIR / f'seed{seed:03d}_f{frame_i:04d}.png'
        Image.fromarray(rendered).save(out_path)
        print(f'  saved: {out_path.name}')
        rendered_frames[seed] = rendered

    # ── Grid comparison: GT + all seeds ──────────────────────────────────────
    gt_path = GT_FRAMES_DIR / f'frame_{frame_i:04d}.png'
    gt_img  = Image.open(gt_path).convert('RGB').resize((518, 518))

    all_imgs   = [gt_img] + [Image.fromarray(rendered_frames[s]) for s in seeds]
    all_labels = [f'GT f{frame_i}'] + [f'seed={s}' for s in seeds]

    W, H    = 518, 518
    LABEL_H = 28
    COLS    = 5
    ROWS    = (len(all_imgs) + COLS - 1) // COLS
    grid    = Image.new('RGB', (W * COLS, (H + LABEL_H) * ROWS), (30, 30, 30))
    draw    = ImageDraw.Draw(grid)

    for idx, (img, label) in enumerate(zip(all_imgs, all_labels)):
        col = idx % COLS
        row = idx // COLS
        x   = col * W
        y   = row * (H + LABEL_H)
        grid.paste(img, (x, y + LABEL_H))
        draw.rectangle([x, y, x + W - 1, y + LABEL_H - 1], fill=(30, 30, 30))
        draw.text((x + W // 2, y + LABEL_H // 2), label, fill='white', anchor='mm')

    grid_path = RESULTS_DIR / f'seed_grid_f{frame_i:04d}.png'
    grid.save(grid_path)
    print(f'\nGrid saved: {grid_path}')
    print(f'All done. Log: {RESULTS_DIR}/run.log')


if __name__ == '__main__':
    main()
