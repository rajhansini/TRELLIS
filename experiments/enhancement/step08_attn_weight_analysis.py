"""
Step 08 — MCFM Temporal Attention Weight Analysis

For each frame i in 1..150, extract the attention weights from the MCFM
softmax and average over all 1374 spatial token positions.

v2 (per-position): attn shape (1374, n_window) → mean over 1374 → (n_window,)
v3 (joint):        per-frame summed attn (1374, n_window) → mean over 1374 → (n_window,)

Window C: [t, t+1]        → 2 weights: [w_curr, w_next]
Window D: [t-1, t, t+1]   → 3 weights: [w_prev, w_curr, w_next]

BACKWARD COMPATIBLE: mcfm.py, step07b, all training scripts untouched.

Results per mode: results_attn_{mode}/
  attn_weights.npy   — (150, n_window) float32
  window_frames.json — for each frame, which window indices were used
  attn_analysis.log  — full log
  temporal_weights_{mode}.png  — plot of 3 curves over 150 frames

Usage:
  python step08_attn_weight_analysis.py --mode v2_C
  python step08_attn_weight_analysis.py --mode v2_D
  python step08_attn_weight_analysis.py --mode v3_C
  python step08_attn_weight_analysis.py --mode v3_D
"""

import sys, os, argparse as _ap
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode', type=str, default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_PRE, _ = _pre.parse_known_args()

_RESULTS = _HERE / f'results_attn_{_PRE.mode}'
_RESULTS.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

sys.stdout = _Tee(_RESULTS / 'attn_analysis.log')
sys.stderr = sys.stdout

import json, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from step4_mcfm.mcfm_analyze import mcfm_v2_attn, mcfm_v3_attn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
PRETRAINED    = 'JeffreyXiang/TRELLIS-image-large'
DEVICE        = 'cuda'
N_FRAMES      = 150
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
_DINO_NORM    = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_window(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    else:
        return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(win: list) -> torch.Tensor:
    return torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)


def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)   # (1374, 1024)


def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode', type=str, default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args = parser.parse_args()
    mode = args.mode

    n_window = 2 if mode.endswith('_C') else 3
    if mode.endswith('_C'):
        labels = ['current(t)', 'next(t+1)']
        curr_idx = 0   # current frame is at window position 0
    else:
        labels = ['prev(t-1)', 'current(t)', 'next(t+1)']
        curr_idx = 1   # current frame is at window position 1

    mcfm_fn = mcfm_v2_attn if mode.startswith('v2') else mcfm_v3_attn

    print('=' * 72)
    print('Step 08 — MCFM Temporal Attention Weight Analysis')
    print('=' * 72)
    print(f'  mode      : {mode}')
    print(f'  mcfm      : {"v2 (per-position)" if mode.startswith("v2") else "v3 (joint)"}')
    print(f'  window    : {"C=[t,t+1]" if mode.endswith("_C") else "D=[t-1,t,t+1]"}')
    print(f'  n_frames  : {N_FRAMES}')
    print(f'  results   : {_RESULTS}')
    print()

    # ── Load pipeline ──────────────────────────────────────────────────────────
    print('[LOAD] Loading TRELLIS pipeline (need DINOv2 model)...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)

    # Move everything except DINOv2 to CPU to save VRAM
    for name in list(pipeline.models.keys()):
        if name != 'image_cond_model':
            try: pipeline.models[name].cpu()
            except: pass
    torch.cuda.empty_cache()

    # ── Encode all frames ──────────────────────────────────────────────────────
    print(f'[ENCODE] Encoding {N_FRAMES} frames with DINOv2...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    t0 = time.time()
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 30 == 0:
            print(f'  encoded {i}/{N_FRAMES}  ({time.time()-t0:.1f}s)')
    dino_model.cpu()
    torch.cuda.empty_cache()
    print(f'[ENCODE] Done. ({time.time()-t0:.1f}s)')

    # ── Extract attention weights for all frames ───────────────────────────────
    print(f'\n[ANALYSIS] Extracting MCFM attention weights for all {N_FRAMES} frames...')
    all_attn   = np.zeros((N_FRAMES, n_window), dtype=np.float32)
    window_log = {}

    for i in range(1, N_FRAMES + 1):
        win = get_window(i, mode)
        lam = get_lambda_vec(win)
        tok_d = {idx: {'tokens': raw_tokens[idx].to(DEVICE)} for idx in set(win)}

        with torch.no_grad():
            _, _, attn_w = mcfm_fn(tok_d, win, i, lam)
            # attn_w: (1374, n_window) — average over spatial positions
            mean_w = attn_w.mean(dim=0).cpu().numpy()   # (n_window,)

        all_attn[i - 1] = mean_w
        window_log[i]   = win

        if i % 30 == 0 or i == 1:
            w_str = '  '.join(f'{labels[j]}={mean_w[j]:.4f}' for j in range(n_window))
            print(f'  frame {i:3d}: win={win}  {w_str}')

    torch.cuda.empty_cache()

    # ── Save raw data ──────────────────────────────────────────────────────────
    npy_path = _RESULTS / 'attn_weights.npy'
    np.save(npy_path, all_attn)
    print(f'\n[SAVE] attn_weights.npy saved → shape {all_attn.shape}')

    with open(_RESULTS / 'window_frames.json', 'w') as f:
        json.dump(window_log, f, indent=2)
    print('[SAVE] window_frames.json saved')

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f'\n[STATS] Mean attention weights over all {N_FRAMES} frames:')
    for j, lbl in enumerate(labels):
        print(f'  {lbl}: mean={all_attn[:, j].mean():.4f}  '
              f'min={all_attn[:, j].min():.4f}  max={all_attn[:, j].max():.4f}')

    # ── Plot ───────────────────────────────────────────────────────────────────
    frame_axis = np.arange(1, N_FRAMES + 1)
    colors = ['#E07B54', '#4C9BE8', '#6DBE6D']   # prev=orange, curr=blue, next=green

    if mode.endswith('_C'):
        colors = ['#4C9BE8', '#6DBE6D']   # curr=blue, next=green

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, (lbl, col) in enumerate(zip(labels, colors)):
        ax.plot(frame_axis, all_attn[:, j], label=lbl, color=col, linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Frame index', fontsize=12)
    ax.set_ylabel('Mean attention weight', fontsize=12)
    ax.set_title(f'MCFM {mode} — temporal attention weights over 150 frames', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1, N_FRAMES)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = _RESULTS / f'temporal_weights_{mode}.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'[PLOT] Saved → {plot_path}')

    print(f'\n[DONE] Results in {_RESULTS}')


if __name__ == '__main__':
    main()
