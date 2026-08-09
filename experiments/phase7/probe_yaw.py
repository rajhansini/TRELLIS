"""
Quick yaw probe for phase7 Gaussian cache.
Renders frame 1 at 8 yaw angles and saves a grid next to the GT front.png.
Output: experiments/results/phase7/yaw_probe.png
Run:
    python experiments/phase7/probe_yaw.py
"""
import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis.utils import render_utils
from trellis.representations import Gaussian

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = REPO_ROOT / 'experiments' / 'results' / 'phase7' / 'gaussian_cache.npz'
GT_PATH    = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                  '/mvadaptorresults/trellis_150_frames/frame_0001/renders/front.png')
OUT_PATH   = REPO_ROOT / 'experiments' / 'results' / 'phase7' / 'yaw_probe.png'
RENDER_RES = 512
GAUSS_KEYS = ['_xyz', '_scaling', '_rotation', '_features_dc', '_opacity']


def get_rep_config():
    snap = Path('/net/scratch/rajhansini/.cache/huggingface/hub'
                '/models--microsoft--TRELLIS-image-large/snapshots')
    if not snap.exists():
        snap = Path.home() / '.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots'
    snaps = list(snap.glob('*'))
    cfg_path = snaps[0] / 'ckpts' / 'slat_dec_gs_swin8_B_64l8gs32_fp16.json'
    with open(cfg_path) as f:
        return json.load(f)['args']['representation_config']


def make_gaussian(params_np, rep_config, device):
    g = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=rep_config['3d_filter_kernel_size'],
        scaling_bias=rep_config['scaling_bias'],
        opacity_bias=rep_config['opacity_bias'],
        scaling_activation=rep_config['scaling_activation'],
    )
    for k in GAUSS_KEYS:
        t = torch.from_numpy(params_np[k].copy()).to(device)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if k == '_scaling':
            t = t.clamp(-10.0, 4.0)
        setattr(g, k, t)
    return g


def render_at_yaw(g, yaw_deg, device):
    yaw = math.radians(yaw_deg)
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [0.0], 2.0, 40.0,
    )
    from trellis.utils import render_utils as ru
    frames = ru.render_frames(
        g, extr, intr,
        options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
        verbose=False,
    )
    return Image.fromarray(frames['color'][0])


def label(img, text):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return out


def main():
    device = torch.device('cuda')
    print('Loading Gaussian cache ...')
    data = np.load(CACHE_PATH)
    params = {k: data[k][74] for k in GAUSS_KEYS}   # frame 75 (mid-animation, has texture)

    rep_config = get_rep_config()
    g = make_gaussian(params, rep_config, device)

    gt = Image.open(GT_PATH).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt = label(gt, 'GT front.png')

    import utils3d.torch as u3d

    fov_t = torch.deg2rad(torch.tensor(40.)).cuda()
    intr_std = u3d.intrinsics_from_fov_xy(fov_t, fov_t)

    def render_custom(eye, up_vec, label_str):
        eye_t = torch.tensor(eye, dtype=torch.float32).cuda()
        up_t  = torch.tensor(up_vec, dtype=torch.float32).cuda()
        tgt   = torch.zeros(3).cuda()
        extr  = u3d.extrinsics_look_at(eye_t, tgt, up_t)
        frames = render_utils.render_frames(
            g, [extr], [intr_std],
            options={'resolution': RENDER_RES, 'bg_color': (1, 1, 1)},
            verbose=False,
        )
        img = Image.fromarray(frames['color'][0])
        return label(img, label_str)

    # Teapot body is along Y in TRELLIS space. To see the front (body upright),
    # look from Z or X directions with Y as up.
    cameras = [
        ([0,  0,  2], [0, 1, 0], '+Z  Y-up'),
        ([0,  0, -2], [0, 1, 0], '-Z  Y-up'),
        ([2,  0,  0], [0, 1, 0], '+X  Y-up'),
        ([-2, 0,  0], [0, 1, 0], '-X  Y-up'),
        ([0,  2,  0], [0, 0, 1], '+Y  Z-up (default)'),
        ([1.5, 0, 1.5], [0, 1, 0], '+XZ Y-up'),
        ([-1.5,0, 1.5], [0, 1, 0], '-XZ Y-up'),
        ([0,  1,  2], [0, 1, 0], '+YZ Y-up'),
    ]

    panels = [gt]
    for eye, up_vec, lbl in cameras:
        print(f'  {lbl}')
        panels.append(render_custom(eye, up_vec, lbl))

    total_w = RENDER_RES * len(panels)
    grid = Image.new('RGB', (total_w, RENDER_RES), (200, 200, 200))
    for i, p in enumerate(panels):
        grid.paste(p, (i * RENDER_RES, 0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT_PATH)
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
