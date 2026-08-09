"""
Verify Step 6: Local Attention Enhancement.

Run from this directory:
  cd .../step6_attn_enhancement
  SPCONV_ALGO=native ATTN_BACKEND=xformers python verify_step6.py 2>&1 | tee ../results/verify_step6.log
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

from step1_input_prep.input_prep       import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from enhancement                        import build_enhancement_matrix, enhanced_attention_ctx

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
flow_model.eval()
for p in flow_model.parameters():
    p.requires_grad_(False)

# ── Voxel structure (G_S) ─────────────────────────────────────────────────────
print('\nGetting voxel structure...')
img_75 = Image.open(GT_FRAME_75).convert('RGB')
cond   = pipeline.get_cond([img_75])
torch.manual_seed(NOISE_SEED)
coords = pipeline.sample_sparse_structure(cond, num_samples=1)
N_vox  = coords.shape[0]
print(f'  N_vox (input): {N_vox}')

# ── Steps 1–4 ────────────────────────────────────────────────────────────────
t_val, k = 0.5, 3
print(f'\nSteps 1-4  t={t_val}, k={k}...')
frame_idx, window_indices, window_frames, _ = prepare_input(t=t_val, k=k)
dino        = load_dino(device)
tokens      = encode_frames(window_indices, window_frames, dino, device)
projector   = FrameWeightProjector(k=k).to(device)
lambda_vec, _ = get_frame_weights(t_val, frame_idx, window_indices, tokens, projector)
K_hat, _    = run_mcfm('v1', tokens, window_indices, frame_idx, lambda_vec)
cond_gl     = K_hat.unsqueeze(0).to(device)
print(f'  lambda_vec (center λ_k={lambda_vec[k].item():.4f}): {[f"{v.item():.3f}" for v in lambda_vec]}')
print(f'  K_hat shape: {tuple(K_hat.shape)}')

# ── Step 5: alignment ─────────────────────────────────────────────────────────
print('\nStep 5: extracting alignment...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)
_, _, voxel_to_token = extract_alignment(flow_model, sparse_x, cond_gl, device)
N_vox_ca = voxel_to_token.shape[0]
print(f'  N_vox_ca (cross-attn level): {N_vox_ca}')

# ── Step 6: build E ───────────────────────────────────────────────────────────
print('\nStep 6: building enhancement matrix E...')
E = build_enhancement_matrix(voxel_to_token, lambda_vec, k=k, N_ctx=1374)
print(f'  E shape    : {tuple(E.shape)}')
print(f'  E non-zeros: {(E != 0).sum().item()}  (expected {N_vox_ca})')
print(f'  E unique values: {E.unique().tolist()}  (expected [0.0, λ_center])')

assert E.shape == (N_vox_ca, 1374), f"E shape wrong: {E.shape}"
assert (E != 0).sum().item() == N_vox_ca, "Wrong number of non-zero entries in E"

lambda_center = lambda_vec[k].item()
nonzero_vals  = E[E != 0].unique().tolist()
assert len(nonzero_vals) == 1 and abs(nonzero_vals[0] - lambda_center) < 1e-5, \
    f"E non-zero value wrong: {nonzero_vals} vs expected {lambda_center}"
print('  E structure: OK')

# ── Baseline forward (no enhancement) ────────────────────────────────────────
print('\nBaseline G_L forward (no enhancement)...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)
t_tensor = torch.tensor([500.0], device=device)

with torch.no_grad():
    baseline_out = flow_model(sparse_x, t_tensor, cond_gl)

assert baseline_out.feats.shape == (N_vox, flow_model.out_channels), \
    f"Baseline output shape wrong: {baseline_out.feats.shape}"
print(f'  baseline output: feats={tuple(baseline_out.feats.shape)}  OK')

# ── Enhanced forward ──────────────────────────────────────────────────────────
print('\nEnhanced G_L forward (with E)...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)

with torch.no_grad():
    with enhanced_attention_ctx(flow_model, E):
        enhanced_out = flow_model(sparse_x, t_tensor, cond_gl)

assert enhanced_out.feats.shape == (N_vox, flow_model.out_channels), \
    f"Enhanced output shape wrong: {enhanced_out.feats.shape}"
print(f'  enhanced output: feats={tuple(enhanced_out.feats.shape)}  OK')

# ── Verify outputs differ (enhancement changes the result) ───────────────────
diff = (enhanced_out.feats.float() - baseline_out.feats.float()).abs().mean().item()
print(f'\nMean abs diff (enhanced vs baseline): {diff:.6f}  (should be > 0)')
assert diff > 0, "Enhanced and baseline outputs are identical — E had no effect"

# ── Verify patching was reversed ─────────────────────────────────────────────
print('\nVerifying cross_attn.forward restored after context manager...')
for i, block in enumerate(flow_model.blocks):
    if hasattr(block, 'cross_attn'):
        assert not hasattr(block.cross_attn.forward, '__name__') or \
               block.cross_attn.forward.__name__ != 'patched', \
               f"Block {i} cross_attn forward was NOT restored"
print('  Patch restoration: OK')

print('\nStep 6 verified.')
