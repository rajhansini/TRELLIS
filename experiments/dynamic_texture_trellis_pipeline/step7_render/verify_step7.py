"""
Verify Step 7: Full pipeline — Steps 1-6.5 + G_L denoising + decode + nvdiffrast render.

Checks:
  1. rendered_image shape correct: (3, 518, 518)
  2. No NaNs/Infs in rendered output
  3. alpha.grad non-zero after backward (gradient flows through LoRA)
  4. LoRA A/B params have gradients

Run from this directory:
  cd .../step7_render
  SPCONV_ALGO=native ATTN_BACKEND=xformers python verify_step7.py 2>&1 | tee ../results/verify_step7.log
"""

import os, sys, math
os.environ['SPCONV_ALGO']         = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']            = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']     = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.renderers.mesh_renderer import MeshRenderer

from step1_input_prep.input_prep       import prepare_input
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from step6_attn_enhancement.enhancement import build_enhancement_matrix
from step6_5_lora.lora                  import insert_lora, count_trainable
from step6_5_lora.ray_attention         import dual_path_ctx

PRETRAINED  = 'microsoft/TRELLIS-image-large'
NOISE_SEED  = 42
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

RENDER_RES  = 518
device      = torch.device('cuda')
print(f'Device: {device}')

# ── Front-view camera ─────────────────────────────────────────────────────────
# TRELLIS objects: vertices in ~[-0.5,0.5]^3 (confirmed from decode).
# intrinsics_to_projection expects NORMALIZED values (divided by resolution):
#   fx_norm = 1 / (2 * tan(fov/2)),  cx_norm = 0.5
# Camera at world z=-2 looking in +Z; z_cam = z_world + 2 → object at z_world=0 → z_cam=2 > 0.
fov_deg = 40.0
fx_n = fy_n = 1.0 / (2.0 * math.tan(math.radians(fov_deg / 2)))
INTRINSICS  = torch.tensor([[fx_n, 0., 0.5], [0., fy_n, 0.5], [0., 0., 1.]],
                             dtype=torch.float32, device=device)
EXTRINSICS  = torch.eye(4, dtype=torch.float32, device=device)
EXTRINSICS[2, 3] = 2.0    # z_cam = z_world + 2; objects at z=0 → z_cam=2 > near=0.1

# ── Load pipeline ─────────────────────────────────────────────────────────────
print('\nLoading TRELLIS pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(device)
flow_model = pipeline.models['slat_flow_model']

# ── Insert LoRA ───────────────────────────────────────────────────────────────
print('Inserting LoRA (rank=4)...')
lora_blocks, alpha = insert_lora(flow_model, rank=4)
lora_blocks = lora_blocks.to(device)
alpha       = nn.Parameter(alpha.data.to(device))
print(f'  Trainable params: {count_trainable(lora_blocks, alpha):,}')

# ── Steps 1-4: MCFM v2 ───────────────────────────────────────────────────────
t_val, k = 0.5, 3
print(f'\nSteps 1-4  t={t_val}, k={k}, MCFM=v2...')
frame_idx, window_indices, window_frames, _ = prepare_input(t=t_val, k=k)
dino          = load_dino(device)
tokens        = encode_frames(window_indices, window_frames, dino, device)
projector     = FrameWeightProjector(k=k).to(device)
lambda_vec, _ = get_frame_weights(t_val, frame_idx, window_indices, tokens, projector)
K_hat, _      = run_mcfm('v2', tokens, window_indices, frame_idx, lambda_vec)
cond_gl       = K_hat.unsqueeze(0).to(device)

# K_pooled: temporal mean for LoRA path B
unique_window = list(dict.fromkeys(window_indices))
K_pooled = torch.stack([tokens[idx]['tokens'] for idx in unique_window]).mean(0).to(device)
print(f'  K_hat   : {tuple(K_hat.shape)}')
print(f'  K_pooled: {tuple(K_pooled.shape)}')

# ── Voxel structure (G_S) ─────────────────────────────────────────────────────
print('\nGetting voxel structure (G_S)...')
img_75 = Image.open(GT_FRAME_75).convert('RGB')
cond   = pipeline.get_cond([img_75])
torch.manual_seed(NOISE_SEED)
coords = pipeline.sample_sparse_structure(cond, num_samples=1)
N_vox  = coords.shape[0]
print(f'  N_vox: {N_vox}')

# ── Step 5: alignment ─────────────────────────────────────────────────────────
print('\nStep 5: extracting cross-attn alignment...')
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x0 = sp.SparseTensor(feats=noise, coords=coords)
with torch.no_grad():
    _, _, voxel_to_token = extract_alignment(flow_model, sparse_x0, cond_gl, device)
N_vox_ca = voxel_to_token.shape[0]
print(f'  N_vox_ca: {N_vox_ca}')

# ── Step 6: enhancement matrix E ─────────────────────────────────────────────
print('\nStep 6: building E...')
E = build_enhancement_matrix(voxel_to_token, lambda_vec, k=k, N_ctx=1374).to(device)
print(f'  E shape: {tuple(E.shape)}  non-zeros: {(E != 0).sum().item()}')

# ── Step 7a: full G_L denoising (no_grad, for shape/NaN check) ───────────────
print('\nStep 7a: G_L 25-step denoising + decode + render (no_grad)...')
neg_cond   = torch.zeros_like(cond_gl)
cond_dict  = {'cond': cond_gl, 'neg_cond': neg_cond}

flow_model.eval()
with torch.no_grad():
    with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
        slat = pipeline.sample_slat(cond_dict, coords)

print(f'  SLaT feats: {tuple(slat.feats.shape)}')
assert not torch.isnan(slat.feats).any(),  "NaN in SLaT feats"
assert not torch.isinf(slat.feats).any(),  "Inf in SLaT feats"
print('  SLaT: no NaNs/Infs OK')

# ── Decode mesh ───────────────────────────────────────────────────────────────
print('\nDecoding SLaT → mesh...')
with torch.no_grad():
    decoded = pipeline.decode_slat(slat, ['mesh'])
mesh = decoded['mesh'][0]
print(f'  vertices : {tuple(mesh.vertices.shape)}')
print(f'  faces    : {tuple(mesh.faces.shape)}')
print(f'  vertex_attrs: {tuple(mesh.vertex_attrs.shape)}')
v = mesh.vertices
print(f'  vertex XYZ range: x=[{v[:,0].min():.3f},{v[:,0].max():.3f}]  '
      f'y=[{v[:,1].min():.3f},{v[:,1].max():.3f}]  '
      f'z=[{v[:,2].min():.3f},{v[:,2].max():.3f}]')
assert mesh.success, "Mesh extraction failed (zero vertices or faces)"

# ── Render with nvdiffrast ───────────────────────────────────────────────────
print('\nRendering with nvdiffrast...')
renderer = MeshRenderer(
    rendering_options={'resolution': RENDER_RES, 'near': 0.1, 'far': 100.0, 'ssaa': 1},
    device=str(device),
)
result = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color', 'mask'])
color  = result['color']   # (3, H, W)
mask   = result['mask']    # (1, H, W) or (H, W)

print(f'  color shape: {tuple(color.shape)}')
assert color.shape == (3, RENDER_RES, RENDER_RES), f"Wrong shape: {color.shape}"
assert not torch.isnan(color).any(), "NaN in rendered color"
assert not torch.isinf(color).any(), "Inf in rendered color"
assert color.min() >= 0.0 and color.max() <= 1.0 + 1e-4, \
    f"Color out of [0,1]: [{color.min():.4f}, {color.max():.4f}]"
coverage = mask.float().mean().item() * 100
print(f'  color range : [{color.min():.4f}, {color.max():.4f}]')
print(f'  mask coverage: {coverage:.1f}%')
assert coverage > 0.5, f"Mask coverage too low ({coverage:.2f}%) — camera may be misaligned"
print('  Render: OK')

# Save rendered image
img_out = (color.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
out_path = RESULTS_DIR / 'step7_render_frame75.png'
Image.fromarray(img_out).save(out_path)
print(f'  Saved: {out_path}')

# ── Step 7b: gradient check (1 G_L step with grad tracking) ──────────────────
print('\nStep 7b: gradient check (single G_L forward, LoRA active)...')
flow_model.train()
torch.manual_seed(NOISE_SEED)
noise    = torch.randn(N_vox, flow_model.in_channels, device=device)
sparse_x = sp.SparseTensor(feats=noise, coords=coords)
t_tensor = torch.tensor([500.0], device=device)

# Zero any existing grads
for p in lora_blocks.parameters():
    if p.requires_grad:
        p.grad = None
alpha.grad = None

with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
    out_g = flow_model(sparse_x, t_tensor, cond_gl)

loss = out_g.feats.float().sum()
loss.backward()

assert alpha.grad is not None, "alpha has no gradient"
print(f'  alpha.grad = {alpha.grad.item():.4f}  OK')

lora_trainable = [(n, p) for n, p in lora_blocks.named_parameters() if p.requires_grad]
no_grad        = [n for n, p in lora_trainable if p.grad is None]
assert len(no_grad) == 0, f"LoRA params with no grad: {no_grad[:3]}"
print(f'  All {len(lora_trainable)} LoRA A/B params have gradients: OK')

print('\nStep 7 verified.')
print(f'  Rendered frame saved to: {out_path}')
