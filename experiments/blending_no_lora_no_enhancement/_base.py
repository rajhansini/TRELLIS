"""
Shared pipeline utilities for blending_no_lora_no_enhancement experiments.
No experiment logic here — just infrastructure.
"""

import os, sys, math, time

def _setup_logging(log_path):
    """Tee stdout+stderr to a log file alongside the terminal."""
    class _Tee:
        def __init__(self, path):
            self._f = open(path, 'w', buffering=1)
        def write(self, msg):
            sys.__stdout__.write(msg)
            self._f.write(msg)
        def flush(self):
            sys.__stdout__.flush()
            self._f.flush()
    tee = _Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'
_ROOT = _HERE.parent.parent

import sys
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp
from step1_input_prep.input_prep       import load_frame, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino, encode_frames

PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = str(_PIPE / '..' / '..' / '..' /
                  'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')

DEVICE     = torch.device('cuda')
STEPS      = 25
NOISE_SEED = 42
RENDER_RES = 518
RESCALE_T  = 3.0

SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,  1.2039363384246826
], dtype=torch.float32)
SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338
], dtype=torch.float32)

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32, device=DEVICE)


def load_pipeline():
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    print('Sampling voxel structure from frame 75...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}')

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    return pipeline, flow_model, coords, N_vox


def encode_all_frames():
    print(f'Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino = load_dino(DEVICE)
    t0   = time.time()
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    print(f'  Done in {time.time()-t0:.1f}s')
    del dino
    torch.cuda.empty_cache()
    return all_tokens


def denoise(flow_model, noise_sp, cond_gl):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def render_slat(pipeline, slat, renderer):
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    mask  = result['mask'].unsqueeze(0)
    color = result['color'] * mask + (1.0 - mask)
    return (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def make_renderer():
    return MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )


def assemble_video(out_dir: Path):
    import subprocess
    vid = out_dir / 'render.mp4'
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', '25',
        '-i', str(out_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid)
    ], check=True)
    print(f'Video: {vid}')
