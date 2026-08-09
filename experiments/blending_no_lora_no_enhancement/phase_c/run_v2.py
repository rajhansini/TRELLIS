"""
Phase C — v2: per-position temporal attention, window [i, i+1], k=1.
Log: results/v2/run.log  (all stdout+stderr captured)

Usage:
  python run_v2.py --start_frame 77 --frames 1
  python run_v2.py --start_frame 1 --frames 150
"""

import sys, argparse, time, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import argparse as _ap
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--base_seed', type=int, default=42)
_BASE_SEED, _ = _pre.parse_known_args()
_BASE_SEED = _BASE_SEED.base_seed
_seed_sfx  = f'_seed{_BASE_SEED}' if _BASE_SEED != 42 else ''
_OUT_DIR   = RESULTS_DIR / f'v2{_seed_sfx}'
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Redirect ALL stdout+stderr to log file + terminal ─────────────────────────
class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

_tee = _Tee(_OUT_DIR / 'run.log')
sys.stdout = _tee
sys.stderr = _tee

# ── Now safe to import everything (their prints go to log too) ────────────────
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _base import (
    DEVICE, NOISE_SEED, N_FRAMES,
    load_pipeline, encode_all_frames, denoise,
    normalize_slat, render_slat, make_renderer, assemble_video
)
import trellis.modules.sparse as sp

_SCALE = 1024 ** -0.5


def blend_v2(tok_curr, tok_next):
    Q       = tok_curr.unsqueeze(1)                          # (1374, 1, 1024)
    K       = torch.stack([tok_curr, tok_next], dim=1)       # (1374, 2, 1024)
    scores  = torch.bmm(Q, K.transpose(1, 2)) * _SCALE      # (1374, 1, 2)
    attn    = torch.softmax(scores, dim=-1)                   # (1374, 1, 2)
    blended = torch.bmm(attn, K).squeeze(1)                  # (1374, 1024)
    return blended, attn.squeeze(1)                           # (1374,1024), (1374,2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--base_seed', type=int, default=42)
    args = parser.parse_args()

    start     = args.start_frame
    n_frames  = args.frames or N_FRAMES
    base_seed = args.base_seed

    print('=== Phase C  v2  (per-position attention, window [i, i+1]) ===')
    print(f'  no lambda — attention weights computed from dot products')
    print(f'  frames {start}..{start+n_frames-1}')
    print(f'  log: {_OUT_DIR}/run.log')

    pipeline, flow_model, coords, N_vox = load_pipeline()
    print(f'\n  flow_model in_channels : {flow_model.in_channels}')
    print(f'  voxel coords shape     : {tuple(coords.shape)}  N_vox={N_vox}')

    all_tokens = encode_all_frames()
    renderer   = make_renderer()
    flow_model.eval()

    attn_curr_log = []   # track per-frame mean attn on current frame

    for frame_i in range(start, start + n_frames):
        frame_next = min(N_FRAMES, frame_i + 1)
        tok_curr = all_tokens[frame_i   ]['tokens'].to(DEVICE)
        tok_next = all_tokens[frame_next]['tokens'].to(DEVICE)

        # ── WIRE: blend ───────────────────────────────────────────────────────
        K_hat, attn = blend_v2(tok_curr, tok_next)
        cond_gl     = K_hat.unsqueeze(0)
        attn_mean   = attn.mean(dim=0)                       # (2,)  mean over 1374 positions
        curr_dominant = (attn[:, 0] > attn[:, 1]).float()   # (1374,) 1 where curr wins
        pct_curr_wins = curr_dominant.mean().item() * 100

        attn_curr_log.append(attn_mean[0].item())

        print(f'\n=== frame {frame_i:04d}  (curr={frame_i}, next={frame_next}) ===')
        print(f'  [STEP2] tok_curr  shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}  std={tok_curr.float().std():.5f}')
        print(f'  [STEP2] tok_next  shape={tuple(tok_next.shape)}  mean={tok_next.float().mean():.5f}  std={tok_next.float().std():.5f}')
        print(f'  [ATTN]  mean: curr={attn_mean[0]:.4f}  next={attn_mean[1]:.4f}  sum={attn_mean.sum():.4f}')
        print(f'  [ATTN]  curr > next at {pct_curr_wins:.1f}% of 1374 positions  {"✓ curr dominant" if pct_curr_wins > 50 else "✗ WARNING: next dominant"}')
        print(f'  [ATTN]  per-pos curr attn: min={attn[:,0].min():.4f}  max={attn[:,0].max():.4f}  std={attn[:,0].std():.4f}')
        print(f'  [STEP4] K_hat   shape={tuple(K_hat.shape)}  mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
        print(f'  [STEP4] cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}  -> flow model input OK')

        # ── WIRE: noise ───────────────────────────────────────────────────────
        torch.manual_seed(base_seed + frame_i)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )
        print(f'  [STEP5] noise_sp.feats shape={tuple(noise_sp.feats.shape)}  mean={noise_sp.feats.mean():.5f}  std={noise_sp.feats.std():.5f}')

        # ── WIRE: denoise ─────────────────────────────────────────────────────
        t0 = time.time()
        x0 = denoise(flow_model, noise_sp, cond_gl)
        print(f'  [STEP5] x0.feats  shape={tuple(x0.feats.shape)}  mean={x0.feats.float().mean():.5f}  std={x0.feats.float().std():.5f}')

        # ── WIRE: normalize slat ──────────────────────────────────────────────
        slat = normalize_slat(x0)
        del noise_sp, x0
        print(f'  [STEP6] slat.feats shape={tuple(slat.feats.shape)}  mean={slat.feats.float().mean():.5f}  std={slat.feats.float().std():.5f}')

        # ── WIRE: render ──────────────────────────────────────────────────────
        rendered = render_slat(pipeline, slat, renderer)
        del slat, cond_gl, K_hat, tok_curr, tok_next
        torch.cuda.empty_cache()

        print(f'  [STEP7] rendered shape={rendered.shape}  dtype={rendered.dtype}  min={rendered.min()}  max={rendered.max()}')
        print(f'  [DONE]  elapsed={time.time()-t0:.1f}s')

        out_path = _OUT_DIR / f'frame_{frame_i:04d}.png'
        Image.fromarray(rendered).save(out_path)
        print(f'  saved: {out_path.name}')

    if n_frames > 1:
        assemble_video(_OUT_DIR)

    import numpy as np
    print(f'\n=== ATTENTION SUMMARY (phase_c v2, {len(attn_curr_log)} frames) ===')
    print(f'  mean attn[curr] across all frames : {np.mean(attn_curr_log):.4f}')
    print(f'  min  attn[curr]                   : {np.min(attn_curr_log):.4f}')
    print(f'  max  attn[curr]                   : {np.max(attn_curr_log):.4f}')
    print(f'  frames where curr_attn > 0.5      : {sum(x>0.5 for x in attn_curr_log)}/{len(attn_curr_log)}')
    print(f'  (ideal: attn[curr] > 0.5 always, meaning current frame dominates)')
    print(f'\nAll done. Log: {_OUT_DIR}/run.log')


if __name__ == '__main__':
    main()
