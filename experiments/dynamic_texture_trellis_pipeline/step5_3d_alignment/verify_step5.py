"""
Verify Step 5: 3D alignment via G_L cross-attention hooks.

Run from this directory:
  cd .../step5_3d_alignment
  SPCONV_ALGO=native ATTN_BACKEND=xformers python verify_step5.py 2>&1 | tee ../results/verify_step5.log
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp

from step1_input_prep.input_prep    import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from alignment                          import extract_alignment

PRETRAINED = 'microsoft/TRELLIS-image-large'
NOISE_SEED = 42
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')

device = torch.device('cuda')
print(f'Device: {device}')

# ── Load TRELLIS pipeline ──────────────────────────────────────────────────────
print('\nLoading TRELLIS pipeline...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(device)
flow_model = pipeline.models['slat_flow_model']
flow_model.eval()
for p in flow_model.parameters():
    p.requires_grad_(False)

n_cross_attn = sum(1 for b in flow_model.blocks if hasattr(b, 'cross_attn'))
print(f'G_L cross-attn blocks: {n_cross_attn}')
print(f'G_L in_channels      : {flow_model.in_channels}')

# ── Get voxel structure via G_S (run once on frame 75) ────────────────────────
print('\nGetting voxel structure from G_S (frame 75)...')
img_75 = Image.open(GT_FRAME_75).convert('RGB')
cond   = pipeline.get_cond([img_75])   # {'cond': (1,1374,1024), 'neg_cond': zeros}
print(f'  cond shape: {tuple(cond["cond"].shape)}')

torch.manual_seed(NOISE_SEED)
coords = pipeline.sample_sparse_structure(cond, num_samples=1)
N_vox  = coords.shape[0]
print(f'  voxel coords: {tuple(coords.shape)}  N_vox={N_vox}')

# ── Steps 1-4: get blended K_hat, V_hat ───────────────────────────────────────
t_val = 0.5
k     = 3
print(f'\nRunning Steps 1-4  t={t_val}, k={k}...')
frame_idx, window_indices, window_frames, _ = prepare_input(t=t_val, k=k)
print(f'  frame_idx={frame_idx}, window={window_indices}')

print('  Loading DINOv2...')
dino        = load_dino(device)
tokens      = encode_frames(window_indices, window_frames, dino, device)
projector   = FrameWeightProjector(k=k).to(device)
lambda_vec, _ = get_frame_weights(t_val, frame_idx, window_indices, tokens, projector)
K_hat, _    = run_mcfm('v1', tokens, window_indices, frame_idx, lambda_vec)
print(f'  K_hat shape: {tuple(K_hat.shape)}')

# Reshape to TRELLIS cond format: (1, 1374, 1024)
cond_gl = K_hat.unsqueeze(0).to(device)

# ── Step 5: extract alignment ──────────────────────────────────────────────────
print('\nRunning Step 5 (G_L forward + attention hooks)...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)

attn_map, token_to_voxels, voxel_to_token = extract_alignment(
    flow_model, sparse_x, cond_gl, device, threshold=0.1
)

# ── Verify ─────────────────────────────────────────────────────────────────────
# NOTE: G_L's input_blocks downsample voxels before cross-attention.
# N_vox_ca (cross-attn level) != N_vox (input level). Both are logged below.
N_vox_ca = attn_map.shape[0]
print(f'\nInput voxels (G_S)   : {N_vox}')
print(f'Cross-attn voxels    : {N_vox_ca}  (after G_L downsampling)')
print(f'attn_map shape       : {tuple(attn_map.shape)}')
assert attn_map.shape[1] == 1374, f"Wrong token dim: {attn_map.shape[1]}"

row_sums = attn_map.sum(dim=1)
print(f'attn row sums        : min={row_sums.min():.4f}  max={row_sums.max():.4f}  '
      f'mean={row_sums.mean():.4f}  (should all be ≈1.0)')
assert (row_sums - 1.0).abs().max() < 1e-3, \
    f"Attention rows don't sum to 1: max_err={(row_sums-1.0).abs().max():.6f}"

assert len(token_to_voxels) == 1374, \
    f"token_to_voxels should have 1374 entries, got {len(token_to_voxels)}"
total_forward_slots = sum(len(v) for v in token_to_voxels.values())
print(f'token_to_voxels      : 1374 tokens, {total_forward_slots} total forward-aligned slots')

assert voxel_to_token.shape == (N_vox_ca,), f"Wrong shape: {voxel_to_token.shape}"
assert int(voxel_to_token.min()) >= 0 and int(voxel_to_token.max()) < 1374
print(f'voxel_to_token       : shape={tuple(voxel_to_token.shape)}  '
      f'token range=[{int(voxel_to_token.min())}, {int(voxel_to_token.max())}]')

unique_tokens_used = voxel_to_token.unique().numel()
print(f'unique tokens used   : {unique_tokens_used} / 1374')
print(f'all {N_vox_ca} cross-attn voxels assigned : OK')

print('\nStep 5 verified.')
