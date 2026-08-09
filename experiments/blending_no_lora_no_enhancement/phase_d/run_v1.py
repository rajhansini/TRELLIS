"""
Phase D — v1: naive weighted average, window [i-1, i, i+1], k=1.
Runs ALL 7 configurations in one shot. Each gets its own log at results/<tag>/run.log.

D0: lp=0.00  lc=1.00  ln=0.00
D1: lp=0.10  lc=0.80  ln=0.10
D2: lp=0.20  lc=0.60  ln=0.20
D3: lp=0.25  lc=0.50  ln=0.25
D4: lp=0.33  lc=0.33  ln=0.33
D5: lp=0.40  lc=0.20  ln=0.40
D6: lp=0.50  lc=0.00  ln=0.50

Usage:
  python run_v1.py                                   # all configs, all 150 frames
  python run_v1.py --tags D3,D4                      # subset of configs
  python run_v1.py --start_frame 77 --frames 1       # single frame test
"""

import os, sys, argparse, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _base import (
    DEVICE, NOISE_SEED, INTRINSICS, EXTRINSICS, N_FRAMES,
    load_pipeline, encode_all_frames, denoise,
    normalize_slat, render_slat, make_renderer, assemble_video
)
import trellis.modules.sparse as sp

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

EXPERIMENTS = [
    dict(tag='D0', lp=0.00, lc=1.00, ln=0.00),
    dict(tag='D1', lp=0.10, lc=0.80, ln=0.10),
    dict(tag='D2', lp=0.20, lc=0.60, ln=0.20),
    dict(tag='D3', lp=0.25, lc=0.50, ln=0.25),
    dict(tag='D4', lp=0.33, lc=0.33, ln=0.33),
    dict(tag='D5', lp=0.40, lc=0.20, ln=0.40),
    dict(tag='D6', lp=0.50, lc=0.00, ln=0.50),
]


class ExpLogger:
    """Per-experiment logger that writes to both terminal and results/<tag>/run.log."""
    def __init__(self, log_path):
        self._f = open(log_path, 'w', buffering=1)

    def log(self, msg=''):
        line = msg + '\n'
        sys.__stdout__.write(line)
        self._f.write(line)
        self._f.flush()

    def close(self):
        self._f.close()


def log_wires(logger, tag, lp, lc, ln, tok_prev, tok_curr, tok_next, K_hat, cond_gl, frame_i):
    """Log full pipeline wire info for the first frame."""
    logger.log(f'\n--- Wire Check  frame={frame_i}  tag={tag}  lp={lp}  lc={lc}  ln={ln} ---')
    logger.log(f'  tok_prev  shape={tuple(tok_prev.shape)}  mean={tok_prev.float().mean():.5f}  std={tok_prev.float().std():.5f}')
    logger.log(f'  tok_curr  shape={tuple(tok_curr.shape)}  mean={tok_curr.float().mean():.5f}  std={tok_curr.float().std():.5f}')
    logger.log(f'  tok_next  shape={tuple(tok_next.shape)}  mean={tok_next.float().mean():.5f}  std={tok_next.float().std():.5f}')
    logger.log(f'  blend     lp*prev + lc*curr + ln*next  (lp={lp}  lc={lc}  ln={ln}  sum={lp+lc+ln:.3f})')
    logger.log(f'  K_hat     shape={tuple(K_hat.shape)}  mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
    logger.log(f'  cond_gl   shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}')
    diff_prev = (K_hat - tok_prev).float()
    diff_curr = (K_hat - tok_curr).float()
    diff_next = (K_hat - tok_next).float()
    logger.log(f'  K_hat vs tok_prev  diff_mean={diff_prev.mean():.5f}  diff_max={diff_prev.abs().max():.5f}')
    logger.log(f'  K_hat vs tok_curr  diff_mean={diff_curr.mean():.5f}  diff_max={diff_curr.abs().max():.5f}')
    logger.log(f'  K_hat vs tok_next  diff_mean={diff_next.mean():.5f}  diff_max={diff_next.abs().max():.5f}')
    logger.log(f'--- End Wire Check ---\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--base_seed', type=int, default=42)
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.tags:
        wanted = set(args.tags.split(','))
        exps = [e for e in EXPERIMENTS if e['tag'] in wanted]

    start     = args.start_frame
    n_frames  = args.frames or N_FRAMES
    base_seed = args.base_seed
    seed_sfx  = f'_seed{base_seed}' if base_seed != 42 else ''
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loggers = {}
    for e in exps:
        d = RESULTS_DIR / f'{e["tag"]}{seed_sfx}'
        d.mkdir(exist_ok=True)
        loggers[e['tag']] = ExpLogger(d / 'run.log')

    def glog(msg):
        sys.__stdout__.write(msg + '\n')

    glog('=== Phase D  v1  (naive avg, window [i-1, i, i+1]) ===')
    glog(f"  configs   : {[e['tag'] for e in exps]}")
    glog(f'  frames    : {start} .. {start+n_frames-1}')
    glog(f'  base_seed : {base_seed}')
    for e in exps:
        loggers[e['tag']].log(f"=== {e['tag']}  lp={e['lp']}  lc={e['lc']}  ln={e['ln']}  seed={base_seed}  frames={start}..{start+n_frames-1} ===")

    pipeline, flow_model, coords, N_vox = load_pipeline()
    all_tokens = encode_all_frames()
    renderer   = make_renderer()
    flow_model.eval()

    for frame_i in range(start, start + n_frames):
        frame_prev = max(1, frame_i - 1)
        frame_next = min(N_FRAMES, frame_i + 1)
        tok_prev = all_tokens[frame_prev]['tokens'].to(DEVICE)
        tok_curr = all_tokens[frame_i   ]['tokens'].to(DEVICE)
        tok_next = all_tokens[frame_next]['tokens'].to(DEVICE)

        for exp in exps:
            tag = exp['tag']
            lp, lc, ln = exp['lp'], exp['lc'], exp['ln']
            logger = loggers[tag]

            K_hat   = lp * tok_prev + lc * tok_curr + ln * tok_next
            cond_gl = K_hat.unsqueeze(0)

            if frame_i == start:
                log_wires(logger, tag, lp, lc, ln, tok_prev, tok_curr, tok_next, K_hat, cond_gl, frame_i)

            torch.manual_seed(base_seed + frame_i)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )

            t0   = time.time()
            x0   = denoise(flow_model, noise_sp, cond_gl)
            slat = normalize_slat(x0)
            del noise_sp, x0

            if frame_i == start:
                logger.log(f'  slat.feats  shape={tuple(slat.feats.shape)}  mean={slat.feats.float().mean():.5f}  std={slat.feats.float().std():.5f}')

            rendered = render_slat(pipeline, slat, renderer)
            del slat, cond_gl, K_hat
            torch.cuda.empty_cache()

            out_path = RESULTS_DIR / f'{tag}{seed_sfx}' / f'frame_{frame_i:04d}.png'
            Image.fromarray(rendered).save(out_path)

            elapsed = time.time() - t0
            logger.log(f'  frame {frame_i:04d}  elapsed={elapsed:.1f}s  saved={out_path.name}')

        del tok_prev, tok_curr, tok_next
        if frame_i % 10 == 0 or frame_i == start:
            glog(f'  [{frame_i:03d}/{start+n_frames-1}] all configs done')

    if n_frames > 1:
        for exp in exps:
            assemble_video(RESULTS_DIR / f'{exp["tag"]}{seed_sfx}')

    for lg in loggers.values():
        lg.close()

    glog('\nDone. Logs at:')
    for e in exps:
        glog(f'  {RESULTS_DIR / e["tag"] / "run.log"}')


if __name__ == '__main__':
    main()
