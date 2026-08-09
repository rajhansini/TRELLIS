"""
Rebuild slat_cache.npz correctly by running pipeline.sample_slat() for all 150 frames.

The existing slat_cache.npz has features in the wrong scale — it was generated
by an incorrect process. This script regenerates it using the same interpolation
approach as Phase 7, which produces features in the correct decoder-expected space
(sample_slat internally applies slat = slat * std + mean denormalization).

Run on a compute node with GPU:
  srun -p threedle-contrib --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \
      /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python \
      experiments/phase8/rebuild_slat_cache.py
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
PRETRAINED  = 'microsoft/TRELLIS-image-large'
N_FRAMES    = 150
NOISE_SEED  = 42
FLOW_STEPS  = 25
OUT_PATH    = REPO_ROOT / 'experiments' / 'results' / 'phase7' / 'slat_cache.npz'
TOKENS_PATH = REPO_ROOT / 'experiments' / 'results' / 'phase0' / 'tokens.npz'
COORDS_PATH = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq' / 'frame_0001' / 'latent.npz'

print('Loading pipeline...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.cuda()
for m in pipeline.models.values():
    for p in m.parameters():
        p.requires_grad_(False)
print('Pipeline loaded.')

device = torch.device('cuda')

# Load conditioning tokens
tok = np.load(TOKENS_PATH)
T_0   = torch.from_numpy(tok['T_0'].astype(np.float32)).to(device)
T_150 = torch.from_numpy(tok['T_150'].astype(np.float32)).to(device)
print(f'T_0 shape: {T_0.shape}  T_150 shape: {T_150.shape}')

# Load fixed voxel coordinates
coords_np = np.load(COORDS_PATH)['coords'].astype(np.int32)
batch_col  = np.zeros((len(coords_np), 1), dtype=np.int32)
coords     = torch.from_numpy(np.concatenate([batch_col, coords_np], axis=1)).to(device)
N_vox      = coords.shape[0]
print(f'Voxels: {N_vox}')

slats_out = np.zeros((N_FRAMES, N_vox, 8), dtype=np.float32)

print(f'Sampling SLaT for {N_FRAMES} frames (FLOW_STEPS={FLOW_STEPS}, seed={NOISE_SEED})...')
for t in range(1, N_FRAMES + 1):
    alpha  = (t - 1) / (N_FRAMES - 1)
    T_t    = (1.0 - alpha) * T_0 + alpha * T_150
    cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        slat = pipeline.sample_slat(cond_t, coords, sampler_params={'steps': FLOW_STEPS})

    slats_out[t - 1] = slat.feats.float().cpu().numpy()

    if t % 10 == 0 or t == 1:
        feats = slats_out[t - 1]
        print(f'  t={t:03d}/{N_FRAMES}  alpha={alpha:.3f}  '
              f'mean={feats.mean():.3f}  std={feats.std():.3f}  '
              f'min={feats.min():.3f}  max={feats.max():.3f}')

print(f'\nSaving to {OUT_PATH}')
np.savez_compressed(str(OUT_PATH), slats=slats_out, coords=coords_np)
print('Done. slat_cache.npz rebuilt correctly.')
print(f'  shape: {slats_out.shape}  dtype: {slats_out.dtype}')
print(f'  overall: mean={slats_out.mean():.4f}  std={slats_out.std():.4f}  '
      f'min={slats_out.min():.4f}  max={slats_out.max():.4f}')
