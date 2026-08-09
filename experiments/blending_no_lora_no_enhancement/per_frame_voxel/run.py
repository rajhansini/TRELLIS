"""
Experiment: per-frame voxel structure sampling.

ONE change vs baseline:
  - Baseline (C0): voxel structure fixed from frame 75, reused for all frames
  - This:          voxel structure sampled from EACH frame's own image

Everything else identical: 25 steps, lc=1.0 (no blending), same seed, same camera.
Goal: see if per-frame geometry improves single render quality.

Log: results/run.log

Usage:
  python run.py --start_frame 77 --frames 1
  python run.py --start_frame 1  --frames 150
"""

import sys, argparse, time, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

from pathlib import Path
_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Redirect ALL stdout+stderr to log + terminal ──────────────────────────────
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

# ── Imports (all prints captured) ────────────────────────────────────────────
import torch
from PIL import Image

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / 'dynamic_texture_trellis_pipeline'))

from _base import (
    DEVICE, NOISE_SEED, INTRINSICS, EXTRINSICS, N_FRAMES,
    PRETRAINED, GT_FRAMES_DIR,
    encode_all_frames, denoise, normalize_slat, render_slat,
    make_renderer, assemble_video
)
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp
from step1_input_prep.input_prep import load_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None)
    args = parser.parse_args()

    start    = args.start_frame
    n_frames = args.frames or N_FRAMES

    print('=== per_frame_voxel experiment ===')
    print(f'  change : voxel structure sampled from each frame (not fixed frame 75)')
    print(f'  blending: none (lc=1.0, pure current frame)')
    print(f'  steps  : 25  seed: {NOISE_SEED}+frame_i')
    print(f'  frames : {start}..{start+n_frames-1}')
    print(f'  log    : {RESULTS_DIR}/run.log')

    # Load pipeline — keep image_cond_model on GPU (needed per frame for voxel sampling)
    print('\nLoading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    image_cond = pipeline.models['image_cond_model']
    print(f'  image_cond_model device: {next(image_cond.parameters()).device}')
    print(f'  flow_model device      : {next(flow_model.parameters()).device}')

    # Pre-encode all frames with DINOv2
    all_tokens = encode_all_frames()

    renderer   = make_renderer()
    flow_model.eval()

    for frame_i in range(start, start + n_frames):
        print(f'\n=== frame {frame_i:04d} ===')

        # ── WIRE: per-frame voxel structure ───────────────────────────────────
        img_i       = load_frame(frame_i)
        cond_struct = pipeline.get_cond([img_i])
        torch.manual_seed(NOISE_SEED)   # fixed seed for voxel structure
        coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
        N_vox  = coords.shape[0]
        print(f'  [STEP1] voxel structure from frame {frame_i}')
        print(f'  [STEP1] coords shape={tuple(coords.shape)}  N_vox={N_vox}')

        # ── WIRE: tokens (no blending, pure current frame) ───────────────────
        tok_curr = all_tokens[frame_i]['tokens'].to(DEVICE)
        cond_gl  = tok_curr.unsqueeze(0)
        print(f'  [STEP2] tok_curr shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}  std={tok_curr.float().std():.5f}')
        print(f'  [STEP4] cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}  -> flow model input OK')

        # ── WIRE: noise + denoise ─────────────────────────────────────────────
        torch.manual_seed(NOISE_SEED + frame_i)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )
        print(f'  [STEP5] noise_sp.feats shape={tuple(noise_sp.feats.shape)}')

        t0   = time.time()
        x0   = denoise(flow_model, noise_sp, cond_gl)
        print(f'  [STEP5] x0.feats shape={tuple(x0.feats.shape)}  mean={x0.feats.float().mean():.5f}  std={x0.feats.float().std():.5f}')

        # ── WIRE: normalize + render ──────────────────────────────────────────
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
