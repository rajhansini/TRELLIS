"""
Rebuild slat_cache.npz from mvadaptor_seq per-frame SLAT encoder latents.

These latents were created by encode_dynamic_sequence.py which runs the SLAT
encoder on MVAdapter-textured meshes — they encode the actual per-frame lava
texture, unlike the interpolated sample_slat features which have no color.

30 keyframes → interpolated to 150 frames to match the GT video.

Run on compute node:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  SPCONV_ALGO=native ATTN_BACKEND=xformers \
      /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python \
      experiments/phase8/rebuild_slat_from_mvadaptor.py
"""

import os, sys, json
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
MV_SEQ_DIR  = REPO_ROOT / 'data' / 'dynamic_sequences' / 'mvadaptor_seq'
OUT_PATH    = REPO_ROOT / 'experiments' / 'results' / 'phase7' / 'slat_cache.npz'
N_FRAMES    = 150
RENDER_RES  = 512

# ── Load 30 keyframe latents ──────────────────────────────────────────────────
meta = json.load(open(MV_SEQ_DIR / 'metadata.json'))
taus = np.array(meta['tau_values'])   # (30,) from 0.0 to 1.0
frame_names = meta['frame_names']     # ['frame_0001', ..., 'frame_0030']

print(f'Loading {len(frame_names)} keyframe latents from mvadaptor_seq...')
all_feats = []
coords_np = None
for fn in frame_names:
    d = np.load(MV_SEQ_DIR / fn / 'latent.npz')
    all_feats.append(d['feats'].astype(np.float32))   # (7063, 8)
    if coords_np is None:
        coords_np = d['coords'].astype(np.int32)       # (7063, 3)

all_feats = np.stack(all_feats, axis=0)  # (30, 7063, 8)
N_vox     = all_feats.shape[1]
print(f'  feats shape: {all_feats.shape}  coords shape: {coords_np.shape}')
print(f'  feats: mean={all_feats.mean():.4f}  std={all_feats.std():.4f}  '
      f'min={all_feats.min():.4f}  max={all_feats.max():.4f}')

# ── Quick decode test (verify color before building full cache) ────────────────
print('\nDecode test on keyframe 15 (mid-sequence, should have lava color)...')
pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.cuda()
for m in pipeline.models.values():
    for p in m.parameters(): p.requires_grad_(False)

B      = np.zeros((N_vox, 1), dtype=np.int32)
coords = torch.from_numpy(np.concatenate([B, coords_np], axis=1)).cuda()
feats_t = torch.from_numpy(all_feats[14]).cuda()   # frame 15 (0-indexed = 14)
slat = sp.SparseTensor(feats=feats_t, coords=coords)
with torch.no_grad():
    decoded = pipeline.decode_slat(slat, ['gaussian'])
g = decoded['gaussian'][0]
dc = g._features_dc
print(f'  _features_dc: R={dc[:,0,0].mean():.4f} G={dc[:,0,1].mean():.4f} B={dc[:,0,2].mean():.4f}')
if abs(dc[:,0,0].mean().item() - dc[:,0,1].mean().item()) > 0.01:
    print('  ✓ Color detected (R≠G) — mvadaptor latents encode texture correctly')
else:
    print('  ✗ Still grayscale — latents may need denormalization')
    std  = np.array(pipeline.slat_normalization['std'])
    mean = np.array(pipeline.slat_normalization['mean'])
    print(f'  Trying denorm with std={std[:3]} mean={mean[:3]}...')
    feats_dn = torch.from_numpy((all_feats[14] * std + mean)).cuda()
    slat2 = sp.SparseTensor(feats=feats_dn, coords=coords)
    with torch.no_grad():
        decoded2 = pipeline.decode_slat(slat2, ['gaussian'])
    dc2 = decoded2['gaussian'][0]._features_dc
    print(f'  denormed _features_dc: R={dc2[:,0,0].mean():.4f} G={dc2[:,0,1].mean():.4f} B={dc2[:,0,2].mean():.4f}')

# ── Interpolate 30 → 150 frames ───────────────────────────────────────────────
print(f'\nInterpolating {len(taus)} keyframes → {N_FRAMES} frames...')
# tau_values for 150 video frames: 0/(N-1), 1/(N-1), ..., (N-1)/(N-1)
target_taus = np.linspace(0.0, 1.0, N_FRAMES)
slats_out   = np.zeros((N_FRAMES, N_vox, 8), dtype=np.float32)

for i, tau in enumerate(target_taus):
    # Find surrounding keyframe indices
    idx = np.searchsorted(taus, tau, side='right') - 1
    idx = np.clip(idx, 0, len(taus) - 2)
    lo, hi = taus[idx], taus[idx + 1]
    alpha  = (tau - lo) / (hi - lo) if hi > lo else 0.0
    slats_out[i] = (1.0 - alpha) * all_feats[idx] + alpha * all_feats[idx + 1]

    if (i + 1) % 25 == 0 or i == 0:
        print(f'  t={i+1:03d}/{N_FRAMES}  tau={tau:.3f}  '
              f'mean={slats_out[i].mean():.3f}  std={slats_out[i].std():.3f}')

print(f'\nSaving to {OUT_PATH}')
np.savez_compressed(str(OUT_PATH), slats=slats_out, coords=coords_np)
print('Done.')
print(f'  shape: {slats_out.shape}  overall mean={slats_out.mean():.4f}  std={slats_out.std():.4f}')
