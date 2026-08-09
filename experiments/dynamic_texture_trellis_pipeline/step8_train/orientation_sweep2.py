"""
Grid sweep: 4 yaw directions × 6 pitch angles (including negatives and 180°).
User picks the cell that matches the GT.

Columns: FRONT(0°), LEFT(90°), BACK(180°), RIGHT(270°)
Rows:    pitch -60°, -30°, 0° (flat/test_front6), +15°, +30°, +45°
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
DEVICE     = torch.device('cuda')
RENDER_RES = 160
OUT_DIR    = Path('../results')

_fx_n = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)


def make_extrinsics(yaw_deg, pitch_deg):
    """
    yaw=0,  pitch=0  -> eye at (0,-2,0)  = test_front6
    yaw=180,pitch=0  -> eye at (0,+2,0)  = test_front5 direction
    pitch>0          -> camera moves up
    pitch<0          -> camera moves down
    """
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    R = 2.0
    ex = R * math.sin(yaw)   * math.cos(pitch)
    ey = -R * math.cos(yaw)  * math.cos(pitch)
    ez = R * math.sin(pitch)
    eye = torch.tensor([ex, ey, ez], dtype=torch.float64)
    f   = (-eye / eye.norm())
    up  = torch.tensor([0., 0., 1.], dtype=torch.float64)
    rx  = torch.linalg.cross(f, up)
    if rx.norm() < 1e-6:                           # degenerate: camera at top/bottom
        up = torch.tensor([0., 1., 0.], dtype=torch.float64)
        rx = torch.linalg.cross(f, up)
    rx  = rx / rx.norm()
    ry  = torch.linalg.cross(rx, f)
    neg = -eye
    t   = torch.tensor([rx.dot(neg), ry.dot(neg), f.dot(neg)])
    E   = torch.zeros(4, 4, dtype=torch.float32)
    E[0, :3] = rx.float(); E[0, 3] = float(t[0])
    E[1, :3] = ry.float(); E[1, 3] = float(t[1])
    E[2, :3] = f.float();  E[2, 3] = float(t[2])
    E[3, 3]  = 1.
    return E.to(DEVICE)


def render_one(mesh, renderer, yaw_deg, pitch_deg):
    extr = make_extrinsics(yaw_deg, pitch_deg)
    with torch.no_grad():
        out = renderer.render(mesh, extr, INTRINSICS, return_types=['color', 'mask'])
    color = out['color'] * out['mask'].unsqueeze(0) + (1. - out['mask'].unsqueeze(0))
    return (color.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading pipeline (mesh decoder only)...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)
for name in list(pipeline.models.keys()):
    if name != 'slat_decoder_mesh':
        try: pipeline.models[name].cpu()
        except: pass

print('Loading latent...')
npz        = np.load(LATENT_NPZ)
feats      = torch.from_numpy(npz['feats']).float().to(DEVICE)
coords_xyz = torch.from_numpy(npz['coords']).int().to(DEVICE)
batch_col  = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=DEVICE)
coords     = torch.cat([batch_col, coords_xyz], dim=1)
slat       = sp.SparseTensor(feats=feats, coords=coords)

print('Decoding mesh...')
with torch.no_grad():
    mesh = pipeline.decode_slat(slat, ['mesh'])['mesh'][0]

renderer = MeshRenderer(
    rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
)

# ── Config grid ───────────────────────────────────────────────────────────────
# columns: yaw directions
yaws   = [(  0, 'FRONT\n(0°)'),
           ( 90, 'LEFT\n(90°)'),
           (180, 'BACK\n(180°)'),
           (270, 'RIGHT\n(270°)')]

# rows: pitch (elevation) angles
pitches = [(-60, 'P=-60°'),
           (-30, 'P=-30°'),
           (  0, 'P=0° (front6)'),
           ( 15, 'P=+15°'),
           ( 30, 'P=+30°'),
           ( 45, 'P=+45°'),
           ( 60, 'P=+60°')]

n_rows = len(pitches)
n_cols = len(yaws) + 1   # +1 for GT column

print(f'Rendering {len(yaws)*len(pitches)} configs...')
renders = {}
for yaw_val, _  in yaws:
    for pitch_val, _ in pitches:
        renders[(yaw_val, pitch_val)] = render_one(mesh, renderer, yaw_val, pitch_val)
        print(f'  yaw={yaw_val:4d}° pitch={pitch_val:+4d}° done')

# ── Build image grid ──────────────────────────────────────────────────────────
cell    = RENDER_RES
lbl_h   = 30
pad     = 4
gt_img  = np.array(Image.open(GT_FRAME_75).convert('RGB').resize((cell, cell)))

W = n_cols * (cell + pad) + pad
H = n_rows * (cell + lbl_h + pad) + lbl_h + pad   # extra top row for column headers

canvas = Image.new('RGB', (W, H), (230, 230, 230))
draw   = ImageDraw.Draw(canvas)

# Column headers
header_y = pad
col_labels = ['GT'] + [lbl for _, lbl in yaws]
for col_i, col_lbl in enumerate(col_labels):
    x = pad + col_i * (cell + pad)
    draw.text((x + 4, header_y + 4), col_lbl.replace('\n', ' '), fill=(0, 0, 0))

for row_i, (pitch_val, pitch_lbl) in enumerate(pitches):
    y = lbl_h + pad + row_i * (cell + lbl_h + pad)

    # GT column
    canvas.paste(Image.fromarray(gt_img), (pad, y))
    draw.text((pad + 4, y + cell + 4), pitch_lbl, fill=(80, 80, 80))

    # Render columns
    for col_i, (yaw_val, yaw_lbl) in enumerate(yaws):
        x    = pad + (col_i + 1) * (cell + pad)
        img  = renders[(yaw_val, pitch_val)]
        canvas.paste(Image.fromarray(img), (x, y))
        draw.text((x + 4, y + cell + 4),
                  f'Y={yaw_val}° P={pitch_val:+}°', fill=(0, 0, 0))

out_path = OUT_DIR / 'orientation_sweep2.png'
canvas.save(out_path)
print(f'\nSaved: {out_path}')
print(f'SCP:')
print(f'  scp rajhansini@fe02.ai.cs.uchicago.edu:{out_path.resolve()} ~/Downloads/trellis_results/orientation_sweep2.png')
