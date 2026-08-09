"""
Run multiple seeds across multiple frames and produce a comparison grid.

Grid layout:
  rows = frames
  cols = GT | seed=2 | seed=6 | ...

Log: results/compare_seeds.log

Usage:
  python compare_seeds.py --frames 1 25 50 75 100 125 150 --seeds 2 6
"""

import sys, argparse, time, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

from pathlib import Path
_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results' / 'compare'
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

sys.stdout = _Tee(RESULTS_DIR / 'compare_seeds.log')
sys.stderr = sys.stdout

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / 'dynamic_texture_trellis_pipeline'))

from _base import (
    DEVICE, GT_FRAMES_DIR,
    encode_all_frames, denoise, normalize_slat, render_slat,
    make_renderer, load_pipeline
)
import trellis.modules.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, nargs='+',
                        default=[1, 25, 50, 75, 100, 125, 150])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 6])
    args = parser.parse_args()

    frames = args.frames
    seeds  = args.seeds

    print(f'=== compare_seeds ===')
    print(f'  frames : {frames}')
    print(f'  seeds  : {seeds}')
    print(f'  log    : {RESULTS_DIR}/compare_seeds.log')

    pipeline, flow_model, coords, N_vox = load_pipeline()
    all_tokens = encode_all_frames()
    renderer   = make_renderer()
    flow_model.eval()

    # renders[frame_i][seed] = np.ndarray (518,518,3)
    renders = {f: {} for f in frames}

    for frame_i in frames:
        tok_curr = all_tokens[frame_i]['tokens'].to(DEVICE)
        cond_gl  = tok_curr.unsqueeze(0)
        print(f'\n=== frame {frame_i:04d} ===')
        print(f'  tok_curr shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}')

        for seed in seeds:
            print(f'  -- seed {seed} (actual={seed+frame_i}) --')
            torch.manual_seed(seed + frame_i)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )
            t0   = time.time()
            x0   = denoise(flow_model, noise_sp, cond_gl)
            slat = normalize_slat(x0)
            del noise_sp, x0

            print(f'     slat mean={slat.feats.float().mean():.5f}  std={slat.feats.float().std():.5f}')

            rendered = render_slat(pipeline, slat, renderer)
            del slat
            torch.cuda.empty_cache()

            print(f'     rendered min={rendered.min()}  max={rendered.max()}  elapsed={time.time()-t0:.1f}s')

            out = RESULTS_DIR / f'seed{seed:03d}_f{frame_i:04d}.png'
            Image.fromarray(rendered).save(out)
            renders[frame_i][seed] = rendered

        del tok_curr, cond_gl

    # ── Build comparison grid ─────────────────────────────────────────────────
    # rows = frames, cols = GT + each seed
    W, H    = 518, 518
    LABEL_H = 28
    COLS    = 1 + len(seeds)   # GT + seeds
    ROWS    = len(frames)
    grid    = Image.new('RGB', (W * COLS, (H + LABEL_H) * ROWS), (30, 30, 30))
    draw    = ImageDraw.Draw(grid)

    for row, frame_i in enumerate(frames):
        y = row * (H + LABEL_H)

        # GT column
        gt_img = Image.open(GT_FRAMES_DIR / f'frame_{frame_i:04d}.png').convert('RGB').resize((W, H))
        grid.paste(gt_img, (0, y + LABEL_H))
        draw.rectangle([0, y, W - 1, y + LABEL_H - 1], fill=(30, 30, 30))
        draw.text((W // 2, y + LABEL_H // 2), f'GT f{frame_i}', fill='white', anchor='mm')

        # seed columns
        for col, seed in enumerate(seeds, start=1):
            x    = col * W
            pred = Image.fromarray(renders[frame_i][seed])
            grid.paste(pred, (x, y + LABEL_H))
            draw.rectangle([x, y, x + W - 1, y + LABEL_H - 1], fill=(50, 30, 30))
            draw.text((x + W // 2, y + LABEL_H // 2), f'seed={seed} f{frame_i}',
                      fill='white', anchor='mm')

    grid_path = RESULTS_DIR / f'comparison_seeds{"_".join(str(s) for s in seeds)}.png'
    grid.save(grid_path)
    print(f'\nGrid saved: {grid_path}')
    print(f'All done. Log: {RESULTS_DIR}/compare_seeds.log')


if __name__ == '__main__':
    main()
