"""Verify Step 3: frame weight projector. Run on a GPU node."""

import sys
sys.path.insert(0, '..')

import torch
from step1_input_prep.input_prep import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from frame_weights import FrameWeightProjector, get_frame_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Step 1
t = 0.5
frame_idx, window_indices, window_frames, mesh_path = prepare_input(t=t, k=3)
print(f'frame_idx={frame_idx}, window={window_indices}')

# Step 2
print('Loading DINOv2...')
dino = load_dino(device)
tokens = encode_frames(window_indices, window_frames, dino, device)
print('Encoded.')

# Step 3
for k in [1, 3, 5]:
    frame_idx, window_indices, window_frames, _ = prepare_input(t=t, k=k)
    tokens_k = encode_frames(window_indices, window_frames, dino, device)

    projector = FrameWeightProjector(k=k).to(device)
    lambda_vec, lambda_dict = get_frame_weights(t, frame_idx, window_indices, tokens_k, projector)

    assert lambda_vec.shape == (2*k+1,), f"Wrong shape: {lambda_vec.shape}"
    assert abs(lambda_vec.sum().item() - 1.0) < 1e-5, f"Weights don't sum to 1: {lambda_vec.sum()}"
    assert len(lambda_dict) == 2*k+1

    print(f'\nk={k}:')
    print(f'  lambda_vec shape : {lambda_vec.shape}')
    print(f'  sum              : {lambda_vec.sum().item():.6f}')
    print(f'  weights          : {[f"{v.item():.4f}" for v in lambda_vec]}')
    print(f'  lambda_dict keys : {list(lambda_dict.keys())}')

print('\nStep 3 verified.')
