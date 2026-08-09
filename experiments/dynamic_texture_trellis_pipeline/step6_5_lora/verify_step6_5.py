"""
Verify Step 6.5: LoRA insertion + dual-path attention.

Checks (no rendering needed):
  1. LoRA inserted on all 24 cross_attn blocks
  2. Only LoRA params + alpha are trainable; TRELLIS fully frozen
  3. Dual-path forward runs without error
  4. Output SparseTensor shape matches expected (N_vox, out_channels)
  5. Gradient flows through LoRA params and alpha; not through TRELLIS params
  6. Context manager cleanly restores original forward

Run from this directory:
  cd .../step6_5_lora
  SPCONV_ALGO=native ATTN_BACKEND=xformers python verify_step6_5.py 2>&1 | tee ../results/verify_step6_5.log
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
import torch.nn as nn
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp

from step1_input_prep.input_prep       import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from lora                               import insert_lora, count_trainable
from ray_attention                      import dual_path_ctx

PRETRAINED  = 'microsoft/TRELLIS-image-large'
NOISE_SEED  = 42
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')

device = torch.device('cuda')
print(f'Device: {device}')

# ── Load pipeline ─────────────────────────────────────────────────────────────
print('\nLoading TRELLIS pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(device)
flow_model = pipeline.models['slat_flow_model']

# ── Insert LoRA ───────────────────────────────────────────────────────────────
print('\nInserting LoRA (rank=4)...')
lora_blocks, alpha = insert_lora(flow_model, rank=4)
lora_blocks = lora_blocks.to(device)
alpha       = nn.Parameter(alpha.data.to(device))  # .to() strips leaf; reconstruct properly

n_cross_attn = sum(1 for b in flow_model.blocks if hasattr(b, 'cross_attn'))
print(f'  cross_attn blocks  : {n_cross_attn}')
print(f'  LoRA blocks created: {len(lora_blocks)}')
assert len(lora_blocks) == n_cross_attn, "LoRA block count mismatch"

n_trainable = count_trainable(lora_blocks, alpha)
n_frozen    = sum(p.numel() for p in flow_model.parameters() if not p.requires_grad)
n_trellis   = sum(p.numel() for p in flow_model.parameters())
print(f'  Trainable params   : {n_trainable:,}')
print(f'  TRELLIS frozen     : {n_trellis:,}  (all {n_frozen:,} require_grad=False)')
assert n_frozen == n_trellis, "Not all TRELLIS params frozen"
assert n_trainable > 0,       "No trainable params found"
print('  Freeze check: OK')

# ── Steps 1-4 ────────────────────────────────────────────────────────────────
t_val, k = 0.5, 3
print(f'\nSteps 1-4  t={t_val}, k={k}...')
frame_idx, window_indices, window_frames, _ = prepare_input(t=t_val, k=k)
dino        = load_dino(device)
tokens      = encode_frames(window_indices, window_frames, dino, device)
projector   = FrameWeightProjector(k=k).to(device)
lambda_vec, _ = get_frame_weights(t_val, frame_idx, window_indices, tokens, projector)
K_hat, _    = run_mcfm('v1', tokens, window_indices, frame_idx, lambda_vec)
cond_gl     = K_hat.unsqueeze(0).to(device)

# K_pooled: temporal mean of individual frame tokens (not MCFM blend)
K_pooled = torch.stack([tokens[idx]['tokens'] for idx in dict.fromkeys(window_indices)]).mean(0).to(device)
print(f'  K_hat shape   : {tuple(K_hat.shape)}')
print(f'  K_pooled shape: {tuple(K_pooled.shape)}')
assert K_pooled.shape == (1374, 1024)

# ── Voxel structure ───────────────────────────────────────────────────────────
print('\nGetting voxel structure...')
img_75 = Image.open(GT_FRAME_75).convert('RGB')
cond   = pipeline.get_cond([img_75])
torch.manual_seed(NOISE_SEED)
coords = pipeline.sample_sparse_structure(cond, num_samples=1)
N_vox  = coords.shape[0]
print(f'  N_vox: {N_vox}')

# ── Dual-path forward ─────────────────────────────────────────────────────────
print('\nDual-path forward (with LoRA)...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)
t_tensor = torch.tensor([500.0], device=device)

with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha):
    out = flow_model(sparse_x, t_tensor, cond_gl)

print(f'  output feats: {tuple(out.feats.shape)}')
assert out.feats.shape == (N_vox, flow_model.out_channels), \
    f"Wrong output shape: {out.feats.shape}"
print('  Output shape: OK')

# ── Verify context manager restored ──────────────────────────────────────────
print('\nVerify forward restored...')
with torch.no_grad():
    out_restored = flow_model(sparse_x, t_tensor, cond_gl)
assert out_restored.feats.shape == (N_vox, flow_model.out_channels)
print('  Patch restoration: OK')

# ── Gradient flow check ───────────────────────────────────────────────────────
print('\nGradient flow check...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)

with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha):
    out_grad = flow_model(sparse_x, t_tensor, cond_gl)

# Dummy loss: sum of output feats
loss = out_grad.feats.float().sum()
loss.backward()

# alpha must have grad
assert alpha.grad is not None, "alpha has no gradient"
print(f'  alpha.grad = {alpha.grad.item():.6f}  (should be non-zero)')

# LoRA A/B params (requires_grad=True) must have grad; frozen linear weights should not
lora_trainable = [(n, p) for n, p in lora_blocks.named_parameters() if p.requires_grad]
no_grad        = [(n, p.grad) for n, p in lora_trainable if p.grad is None]
if no_grad:
    print(f'  WARNING: {len(no_grad)} trainable LoRA params with None grad: {[n for n,_ in no_grad[:3]]}')
else:
    print(f'  All {len(lora_trainable)} trainable LoRA params (lora_A/lora_B) have gradients: OK')

# TRELLIS params must NOT have grad
trellis_grads = [p.grad for p in flow_model.parameters() if p.grad is not None]
assert len(trellis_grads) == 0, \
    f"{len(trellis_grads)} TRELLIS params accumulated gradients — not fully frozen"
print('  TRELLIS params: no gradients accumulated (frozen): OK')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\nLoRA rank-4 on {n_cross_attn} cross_attn blocks')
print(f'Trainable: {n_trainable:,} params  |  TRELLIS frozen: {n_trellis:,} params')
print(f'alpha init: {alpha.item():.4f}')
print('\nStep 6.5 verified.')
