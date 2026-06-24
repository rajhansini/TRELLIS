"""
Quick camera probe — renders frame_0001 at 8 candidate cameras and saves a grid.
Run this ONCE to confirm which camera matches the GT front.png before running all 150 frames.

Output: experiments/results/phase0/camera_probe.png
        Left column: GT front.png
        Right columns: TRELLIS render at each candidate camera
"""
import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import utils3d

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew/mvadaptorresults/trellis_150_frames')
OUT_DIR    = Path(__file__).parent.parent / 'results' / 'phase0'
PRETRAINED = 'microsoft/TRELLIS-image-large'
RENDER_RES = 512

# Candidates: (label, eye_xyz_zup, look_at_zup, fov_deg)
CANDIDATES = [
    ('yaw=0   fov40 r2.0',  [0,  2.0, 0],   [0, 0, 0],    40),
    ('yaw=0   fov60 r1.5',  [0,  1.5, 0],   [0, 0, 0.25], 60),
    ('yaw=pi  fov40 r2.0',  [0, -2.0, 0],   [0, 0, 0],    40),
    ('yaw=pi  fov60 r1.5',  [0, -1.5, 0],   [0, 0, 0.25], 60),
    ('yaw=pi2 fov40 r2.0',  [2.0, 0,  0],   [0, 0, 0],    40),
    ('yaw=pi2 fov60 r1.5',  [1.5, 0,  0],   [0, 0, 0.25], 60),
    ('yaw=3pi2 fov40 r2.0', [-2.0, 0, 0],   [0, 0, 0],    40),
    ('yaw=3pi2 fov60 r1.5', [-1.5, 0, 0],   [0, 0, 0.25], 60),
]


def build_camera(eye, target, fov_deg, device):
    eye_t    = torch.tensor(eye,    dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    up_t     = torch.tensor([0., 0., 1.], device=device)
    extr = utils3d.torch.extrinsics_look_at(eye_t, target_t, up_t)
    while extr.dim() > 2: extr = extr[0]
    fov  = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    while intr.dim() > 2: intr = intr[0]
    return extr, intr


def label_img(arr, text):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, RENDER_RES, 20], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(img)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading pipeline...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device

    # Encode frame 1 front view
    img_path = FRAMES_DIR / 'frame_0001' / 'renders' / 'front.png'
    img = Image.open(img_path).convert('RGB')
    T0 = pipeline.encode_image([img])

    # Fixed coords from frame 1 latent
    data = np.load(Path(__file__).parent.parent.parent /
                   'data/dynamic_sequences/trellis_seq/frame_0001/latent.npz')
    coords = data['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    coords4 = torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)

    cond = {'cond': T0, 'neg_cond': torch.zeros_like(T0)}
    torch.manual_seed(42)
    with torch.no_grad():
        slat = pipeline.sample_slat(cond, coords4, sampler_params={'steps': 25})
        decoded = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = decoded['gaussian'][0]

    # GT image
    gt = np.array(Image.open(img_path).convert('RGB').resize((RENDER_RES, RENDER_RES)))
    gt = label_img(gt, 'GT front.png')

    # Render each candidate
    cols = [gt]
    for label, eye, target, fov in CANDIDATES:
        extr, intr = build_camera(eye, target, fov, device)
        frames = render_utils.render_frames(
            gaussian, [extr], [intr],
            options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
            verbose=False,
        )
        arr = label_img(frames['color'][0], label)
        cols.append(arr)
        print(f'  done: {label}')

    grid = np.concatenate(cols, axis=1)
    out_path = OUT_DIR / 'camera_probe.png'
    Image.fromarray(grid).save(out_path)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
