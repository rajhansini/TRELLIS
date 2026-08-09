"""
Quick orientation test: decode one frame and render with the current EXTRINSICS.
Saves test_render.png and test_comparison.png alongside GT frame 75.
No training — just verifies the camera gives the right view direction.
"""
import os, sys, math
import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers import MeshRenderer

PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
DEVICE      = torch.device('cuda')
RENDER_RES  = 518
NOISE_SEED  = 42

_fx_n = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.tensor([
    [-1.,  0., 0., 0.],
    [ 0.,  0., 1., 0.],
    [ 0., -1., 0., 2.],
    [ 0.,  0., 0., 1.],
], dtype=torch.float32, device=DEVICE)

print('Loading pipeline (need decoder only)...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)
# Keep only mesh decoder on GPU
for name in list(pipeline.models.keys()):
    if name != 'slat_decoder_mesh':
        try: pipeline.models[name].cpu()
        except: pass

# Load pre-computed SLaT from latent.npz (frame 1 — just need any valid latent for orientation test)
LATENT_NPZ = ('/net/projects/ranalab/rajhansini/TRELLIS/data'
              '/dynamic_sequences/trellis_seq/frame_0001/latent.npz')

import numpy as np
from trellis.modules import sparse as sp

print('Loading latent from npz...')
npz    = np.load(LATENT_NPZ)
feats  = torch.from_numpy(npz['feats']).float().to(DEVICE)
coords_xyz = torch.from_numpy(npz['coords']).int().to(DEVICE)
# SparseTensor needs (N, 4) coords: [batch_idx, x, y, z]
batch_col  = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=DEVICE)
coords     = torch.cat([batch_col, coords_xyz], dim=1)
slat       = sp.SparseTensor(feats=feats, coords=coords)
print(f'  N_vox: {coords.shape[0]}  feats: {feats.shape}  coords: {coords.shape}')

print('Decoding mesh...')
with torch.no_grad():
    decoded = pipeline.decode_slat(slat, ['mesh'])
    mesh    = decoded['mesh'][0]

print(f'  vertices: {mesh.vertices.shape}  min={mesh.vertices.min().item():.3f}  max={mesh.vertices.max().item():.3f}')

print('Rendering...')
renderer = MeshRenderer(
    rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1},
)
with torch.no_grad():
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color'])
color = result['color']
print(f'  color range: [{color.min().item():.3f}, {color.max().item():.3f}]')

out_dir = Path('../results')
color_np = (color.cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
Image.fromarray(color_np).save(out_dir / 'test_render.png')

gt_np  = np.array(Image.open(GT_FRAME_75).convert('RGB').resize((RENDER_RES, RENDER_RES)))
sbs    = np.concatenate([gt_np, color_np], axis=1)
Image.fromarray(sbs).save(out_dir / 'test_comparison.png')

print(f'\nSaved:')
print(f'  {out_dir}/test_render.png')
print(f'  {out_dir}/test_comparison.png')
print('Done.')
