"""Verify Step 2: DINOv2 encoding (TRELLIS format). Run on a GPU node."""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from step1_input_prep.input_prep import prepare_input
from dino_encoding import load_dino, encode_frames, DINO_DIM, N_TOKENS, IMAGE_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

frame_idx, window_indices, window_frames, _ = prepare_input(t=0.5, k=3)
print(f'frame_idx={frame_idx}, window={window_indices}')

print('Loading DINOv2...')
model = load_dino(device)
tokens_dict = encode_frames(window_indices, window_frames, model, device)

# Shape checks
assert len(tokens_dict) == len(set(window_indices)), "Unexpected token count"
for idx in set(window_indices):
    t = tokens_dict[idx]
    assert t['tokens'].shape == (N_TOKENS, DINO_DIM), \
        f"tokens shape wrong for frame {idx}: {t['tokens'].shape}"
    assert t['cls'].shape == (1, DINO_DIM), \
        f"cls shape wrong for frame {idx}: {t['cls'].shape}"
    assert not torch.isnan(t['tokens']).any(), f"NaN in tokens for frame {idx}"

    # cls must match first row of tokens
    assert torch.allclose(t['cls'], t['tokens'][0:1]), \
        f"cls != tokens[0] for frame {idx}"

    print(f'  frame {idx}: tokens={t["tokens"].shape}  cls={t["cls"].shape}  OK')

# Verify format matches TRELLIS: tokens should be layernorm'd (mean≈0, std≈1 per token)
sample_tok = tokens_dict[frame_idx]['tokens']
mean_per_tok = sample_tok.mean(dim=-1)
print(f'\nLayernorm check (mean per token): min={mean_per_tok.min():.4f}  '
      f'max={mean_per_tok.max():.4f}  (should be ≈0)')

print(f'\nStep 2 verified. {len(tokens_dict)} frames encoded at {IMAGE_SIZE}x{IMAGE_SIZE}, '
      f'{N_TOKENS} tokens of dim {DINO_DIM} each, matching TRELLIS format.')
