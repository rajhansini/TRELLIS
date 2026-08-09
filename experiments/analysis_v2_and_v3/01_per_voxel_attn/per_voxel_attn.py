"""
01 — Per-voxel attention curves for v2 and v3 MCFM.

For 5 sampled spatial token positions, plots w_{t-1}, w_t, w_{t+1}
over all 150 frames to show how each position attends to its window.
The "wiggly" pattern: at frame i, w_t is highest; at frame i+1, the
window shifts so what was w_{t+1} becomes w_t.

BACKWARD COMPATIBLE: reads from existing results, does NOT modify mcfm.py.

Requires GPU (DINOv2 encoding + MCFM).
Output: results/per_voxel_attn_{mode}/

Usage:
  python per_voxel_attn.py --mode v2_C
  python per_voxel_attn.py --mode v3_C
"""

import sys, os, argparse
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from step4_mcfm.mcfm_analyze import mcfm_v2_attn, mcfm_v3_attn

VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED = 'microsoft/TRELLIS-image-large'
DEVICE     = torch.device('cuda')
N_FRAMES   = 150

# Sample 5 spatial positions spread across 1374 tokens
SAMPLE_POSITIONS = [0, 274, 686, 1100, 1373]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def get_window(frame_idx, mode):
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def encode_frame(dino_model, frame_idx):
    img = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    x   = _DINO_NORM(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args = parser.parse_args()
    mode = args.mode

    out_dir = _HERE / 'results' / f'per_voxel_attn_{mode}'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'analysis.log')
    sys.stderr = sys.stdout

    print(f'[PER-VOXEL ATTN] mode={mode}  sample_positions={SAMPLE_POSITIONS}')

    # Load pipeline for DINOv2 only
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)

    print(f'[DINO] Encoding {N_FRAMES} frames...')
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i)
    dino_model.cpu()
    torch.cuda.empty_cache()
    print('  Done.')

    mcfm_fn = mcfm_v2_attn if mode.startswith('v2') else mcfm_v3_attn
    n_win   = 2 if mode.endswith('_C') else 3

    # attn_data: (N_FRAMES, N_SAMPLE, n_win)
    attn_data = np.zeros((N_FRAMES, len(SAMPLE_POSITIONS), n_win), dtype=np.float32)

    print(f'[MCFM] Extracting per-voxel attention ({n_win}-frame window)...')
    for i in range(1, N_FRAMES + 1):
        win  = get_window(i, mode)
        lam  = torch.ones(len(win)) / len(win)
        tok_d = {j: {'tokens': raw_tokens[j].to(DEVICE)} for j in set(win)}
        _, _, attn = mcfm_fn(tok_d, win, i, lam.to(DEVICE))
        # attn: (N_TOKENS, n_win) — extract sampled positions
        for s, pos in enumerate(SAMPLE_POSITIONS):
            attn_data[i - 1, s, :] = attn[pos].cpu().float().numpy()
        if i % 30 == 0 or i == 1:
            print(f'  frame {i}/{N_FRAMES}  sample[2]={attn_data[i-1,2]}')

    np.save(out_dir / f'per_voxel_attn_{mode}.npy', attn_data)
    print(f'[SAVE] {out_dir}/per_voxel_attn_{mode}.npy  shape={attn_data.shape}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    frame_idx = np.arange(1, N_FRAMES + 1)
    win_labels = (['t', 't+1'] if mode.endswith('_C')
                  else ['t-1', 't', 't+1'])
    colors = ['#e06c75', '#61afef', '#98c379']

    fig, axes = plt.subplots(len(SAMPLE_POSITIONS), 1,
                             figsize=(14, 3 * len(SAMPLE_POSITIONS)), sharex=True)
    fig.suptitle(f'Per-token attention weights over 150 frames — {mode}', fontsize=14)

    for s, (ax, pos) in enumerate(zip(axes, SAMPLE_POSITIONS)):
        for w in range(n_win):
            ax.plot(frame_idx, attn_data[:, s, w],
                    label=f'w_{win_labels[w]}', color=colors[w], alpha=0.85, lw=1.2)
        ax.set_ylabel(f'token {pos}', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc='upper right')
        ax.axhline(1.0 / n_win, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame index')
    plt.tight_layout()
    fig.savefig(out_dir / f'per_voxel_attn_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[PLOT] saved per_voxel_attn_{mode}.png')
    print(f'[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
