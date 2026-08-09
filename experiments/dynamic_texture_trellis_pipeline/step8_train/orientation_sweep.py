"""
Render from 8 horizontal angles (every 45°) so user can pick the correct view.
Saves orientation_sweep.png: GT on far left, then 8 angle renders labeled 0°..315°.
"""
import os, sys, math
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers import MeshRenderer
from trellis.modules import sparse as sp

PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
LATENT_NPZ  = ('/net/projects/ranalab/rajhansini/TRELLIS/data'
               '/dynamic_sequences/trellis_seq/frame_0001/latent.npz')
DEVICE      = torch.device('cuda')
RENDER_RES  = 300
OUT_DIR     = Path('../results')

_fx_n = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)


def make_extrinsics(theta_deg):
    """Camera at theta_deg around vertical (world-Z) axis, 0° elevation.
    theta=0  → eye at (0,-2,0)  [current test_front6]
    theta=180 → eye at (0,+2,0) [test_front5 direction]
    """
    theta = math.radians(theta_deg)
    R = 2.0
    eye = torch.tensor([R * math.sin(theta), -R * math.cos(theta), 0.0])
    f   = -eye / eye.norm()           # toward origin
    up  = torch.tensor([0., 0., 1.])
    rx  = torch.linalg.cross(f, up);  rx = rx / rx.norm()   # screen right
    ry  = torch.linalg.cross(rx, f)                          # screen up
    neg_eye = -eye
    t   = torch.tensor([rx.dot(neg_eye), ry.dot(neg_eye), f.dot(neg_eye)])
    E   = torch.zeros(4, 4)
    E[0, :3] = rx;  E[0, 3] = t[0]
    E[1, :3] = ry;  E[1, 3] = t[1]
    E[2, :3] = f;   E[2, 3] = t[2]
    E[3, 3]  = 1.
    return E.to(DEVICE, dtype=torch.float32)


def render_angle(mesh, renderer, theta_deg):
    extr = make_extrinsics(theta_deg)
    with torch.no_grad():
        result = renderer.render(mesh, extr, INTRINSICS, return_types=['color', 'mask'])
    color = result['color']
    mask  = result['mask'].unsqueeze(0)
    color = color * mask + (1.0 - mask)
    return (color.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


print('Loading pipeline (mesh decoder only)...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)
for name in list(pipeline.models.keys()):
    if name != 'slat_decoder_mesh':
        try: pipeline.models[name].cpu()
        except: pass

print('Loading latent from npz...')
npz        = np.load(LATENT_NPZ)
feats      = torch.from_numpy(npz['feats']).float().to(DEVICE)
coords_xyz = torch.from_numpy(npz['coords']).int().to(DEVICE)
batch_col  = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=DEVICE)
coords     = torch.cat([batch_col, coords_xyz], dim=1)
slat       = sp.SparseTensor(feats=feats, coords=coords)

print('Decoding mesh...')
with torch.no_grad():
    decoded = pipeline.decode_slat(slat, ['mesh'])
    mesh    = decoded['mesh'][0]
print(f'  vertices: {mesh.vertices.shape}')

renderer = MeshRenderer(
    rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
)

angles = [0, 45, 90, 135, 180, 225, 270, 315]
print(f'Rendering {len(angles)} angles...')
renders = {}
for theta in angles:
    renders[theta] = render_angle(mesh, renderer, theta)
    print(f'  {theta}° done')

# ── Build grid image ──────────────────────────────────────────────────────────
cell    = RENDER_RES
label_h = 22
pad     = 6

gt_img  = np.array(Image.open(GT_FRAME_75).convert('RGB').resize((cell, cell)))

# Two rows: [GT, 0°, 45°, 90°, 135°] and [GT, 180°, 225°, 270°, 315°]
row1 = [('GT',    gt_img),
        ('0°',    renders[0]),
        ('45°',   renders[45]),
        ('90°',   renders[90]),
        ('135°',  renders[135])]
row2 = [('GT',    gt_img),
        ('180°',  renders[180]),
        ('225°',  renders[225]),
        ('270°',  renders[270]),
        ('315°',  renders[315])]

W = len(row1) * (cell + pad) - pad
H = 2 * (cell + label_h) + pad
canvas = Image.fromarray(np.ones((H, W, 3), dtype=np.uint8) * 240)
draw   = ImageDraw.Draw(canvas)

for row_idx, row in enumerate([row1, row2]):
    y0 = row_idx * (cell + label_h + pad)
    for col_idx, (label, img) in enumerate(row):
        x0 = col_idx * (cell + pad)
        canvas.paste(Image.fromarray(img), (x0, y0))
        draw.text((x0 + cell // 2 - len(label) * 3, y0 + cell + 4), label, fill=(0, 0, 0))

out_path = OUT_DIR / 'orientation_sweep.png'
canvas.save(out_path)
print(f'\nSaved: {out_path}')
print('Run:')
print(f'  scp rajhansini@fe02.ai.cs.uchicago.edu:{out_path.resolve()} ~/Downloads/trellis_results/orientation_sweep.png')
