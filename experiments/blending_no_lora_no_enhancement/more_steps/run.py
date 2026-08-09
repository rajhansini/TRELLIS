"""
Experiment: more denoising steps.

ONE change vs baseline:
  - Baseline (C0): 25 denoising steps, voxel from frame 75
  - This:          50 denoising steps, voxel from frame 75 (all else identical)

Goal: see if more steps improve texture quality / color fidelity.

Log: results/run.log

Usage:
  python run.py --start_frame 77 --frames 1
  python run.py --steps 50 --start_frame 77 --frames 1   (default)
  python run.py --steps 100 --start_frame 77 --frames 1
"""

import sys, argparse, time, os, math
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

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / 'dynamic_texture_trellis_pipeline'))

from _base import (
    DEVICE, NOISE_SEED, INTRINSICS, EXTRINSICS, N_FRAMES,
    PRETRAINED, GT_FRAME_75,
    SLAT_MEAN, SLAT_STD, RESCALE_T,
    encode_all_frames, render_slat, make_renderer, assemble_video,
    load_pipeline
)
import trellis.modules.sparse as sp


def build_t_pairs(steps):
    t_seq  = np.linspace(1, 0, steps + 1)
    t_seq  = RESCALE_T * t_seq / (1 + (RESCALE_T - 1) * t_seq)
    return [(t_seq[i], t_seq[i + 1]) for i in range(steps)]


def denoise_n_steps(flow_model, noise_sp, cond_gl, t_pairs):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in t_pairs:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(x, t_ten, cond_gl)
            x     = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None)
    args = parser.parse_args()

    start    = args.start_frame
    n_frames = args.frames or N_FRAMES
    steps    = args.steps
    t_pairs  = build_t_pairs(steps)

    print(f'=== more_steps experiment ===')
    print(f'  change : denoising steps {steps} (baseline=25)')
    print(f'  blending: none (lc=1.0, pure current frame)')
    print(f'  voxel  : fixed from frame 75 (same as baseline)')
    print(f'  seed   : {NOISE_SEED}+frame_i')
    print(f'  frames : {start}..{start+n_frames-1}')
    print(f'  log    : {RESULTS_DIR}/run.log')

    pipeline, flow_model, coords, N_vox = load_pipeline()
    print(f'\n  flow_model in_channels : {flow_model.in_channels}')
    print(f'  voxel coords shape     : {tuple(coords.shape)}  N_vox={N_vox}')
    print(f'  t_pairs[0]             : {t_pairs[0]}')
    print(f'  t_pairs[-1]            : {t_pairs[-1]}')

    all_tokens = encode_all_frames()
    renderer   = make_renderer()
    flow_model.eval()

    for frame_i in range(start, start + n_frames):
        tok_curr = all_tokens[frame_i]['tokens'].to(DEVICE)
        cond_gl  = tok_curr.unsqueeze(0)

        print(f'\n=== frame {frame_i:04d} ===')
        print(f'  [STEP2] tok_curr shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}  std={tok_curr.float().std():.5f}')
        print(f'  [STEP4] cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}  -> flow model input OK')

        torch.manual_seed(NOISE_SEED + frame_i)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )
        print(f'  [STEP5] noise_sp.feats shape={tuple(noise_sp.feats.shape)}  steps={steps}')

        t0   = time.time()
        x0   = denoise_n_steps(flow_model, noise_sp, cond_gl, t_pairs)
        print(f'  [STEP5] x0.feats shape={tuple(x0.feats.shape)}  mean={x0.feats.float().mean():.5f}  std={x0.feats.float().std():.5f}')

        slat = normalize_slat(x0)
        del noise_sp, x0
        print(f'  [STEP6] slat.feats shape={tuple(slat.feats.shape)}  mean={slat.feats.float().mean():.5f}  std={slat.feats.float().std():.5f}')

        rendered = render_slat(pipeline, slat, renderer)
        del slat, cond_gl, tok_curr
        torch.cuda.empty_cache()

        print(f'  [STEP7] rendered shape={rendered.shape}  min={rendered.min()}  max={rendered.max()}')
        print(f'  [DONE]  elapsed={time.time()-t0:.1f}s')

        out_path = RESULTS_DIR / f'frame_{frame_i:04d}.png'
        Image.fromarray(rendered).save(out_path)
        print(f'  saved: {out_path.name}')

    if n_frames > 1:
        assemble_video(RESULTS_DIR)

    print(f'\nAll done. Log: {RESULTS_DIR}/run.log')


if __name__ == '__main__':
    main()
