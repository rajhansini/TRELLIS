"""Verify Step 4: MCFM. Run on a GPU node."""

import sys
sys.path.insert(0, '..')

import torch
from step1_input_prep.input_prep import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights import FrameWeightProjector, get_frame_weights
from mcfm import run_mcfm, N_TOKENS, DINO_DIM
from step2_dino_encoding.dino_encoding import N_TOKENS as DINO_N_TOKENS
assert N_TOKENS == DINO_N_TOKENS, \
    f"Token count mismatch: mcfm={N_TOKENS} vs dino={DINO_N_TOKENS}"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

t = 0.5
k = 3

# Steps 1-3
frame_idx, window_indices, window_frames, _ = prepare_input(t=t, k=k)
print(f'frame_idx={frame_idx}, window={window_indices}')

print('Loading DINOv2...')
dino = load_dino(device)
tokens = encode_frames(window_indices, window_frames, dino, device)

projector  = FrameWeightProjector(k=k).to(device)
lambda_vec, _ = get_frame_weights(t, frame_idx, window_indices, tokens, projector)
print(f'lambda_vec={[f"{v.item():.3f}" for v in lambda_vec]}')

# Step 4: all versions
for version in ['v1', 'v2', 'v2b', 'v3']:
    K_hat, V_hat = run_mcfm(version, tokens, window_indices, frame_idx, lambda_vec)

    assert K_hat.shape == (N_TOKENS, DINO_DIM), f"[{version}] K_hat shape wrong: {K_hat.shape}"
    assert V_hat.shape == (N_TOKENS, DINO_DIM), f"[{version}] V_hat shape wrong: {V_hat.shape}"
    assert not torch.isnan(K_hat).any(),         f"[{version}] K_hat has NaNs"
    assert not torch.isnan(V_hat).any(),         f"[{version}] V_hat has NaNs"

    print(f'  {version}: K_hat={K_hat.shape}  V_hat={V_hat.shape}  '
          f'K_hat mean={K_hat.mean().item():.4f}  OK')

# Boundary case: t=0.0, k=3 → window has duplicates
print('\nBoundary case t=0.0, k=3:')
frame_idx_b, window_b, frames_b, _ = prepare_input(t=0.0, k=k)
tokens_b   = encode_frames(window_b, frames_b, dino, device)
proj_b     = FrameWeightProjector(k=k).to(device)
lam_b, _   = get_frame_weights(0.0, frame_idx_b, window_b, tokens_b, proj_b)

for version in ['v1', 'v2', 'v2b', 'v3']:
    K_hat, V_hat = run_mcfm(version, tokens_b, window_b, frame_idx_b, lam_b)
    assert K_hat.shape == (N_TOKENS, DINO_DIM)
    assert not torch.isnan(K_hat).any()
    print(f'  {version} boundary: OK')

print('\nStep 4 verified.')
