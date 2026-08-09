"""
render_v4_best.py  — render all 150 frames from the best v4 checkpoint and write mp4.
Usage: python render_v4_best.py [--run-id c85c888f] [--fps 15]
"""
import sys, os, math, gc, argparse
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--run-id',  default='c85c888f')
ap.add_argument('--fps',     type=int, default=15)
ap.add_argument('--res',     type=int, default=518)
args = ap.parse_args()

RUN_DIR  = _HERE / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}'
CKPT     = RUN_DIR / 'lora_ckpts' / 'lora_best.pt'
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
OUT_DIR  = RUN_DIR / 'renders_full'
OUT_DIR.mkdir(exist_ok=True)

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)

# ── nvdiffrast arch guard ─────────────────────────────────────────────────────
def _ensure_nvdiffrast():
    import subprocess, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm{p.major}{p.minor}'
    local    = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)
        return
    if not os.path.isdir(local) and os.environ.get('_NVDIFF_REBUILT') != arch_tag:
        import subprocess
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--target', local,
            '--no-build-isolation', 'git+https://github.com/NVlabs/nvdiffrast.git'
        ], check=True, capture_output=True)
        sys.path.insert(0, local)
        os.environ['_NVDIFF_REBUILT'] = arch_tag

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.modules import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

DEC_OUT_IN_DIM = 96
COLOR_START    = 53
COLOR_END      = 101
COLOR_DIM      = COLOR_END - COLOR_START   # 48
ABS_MIN        = -9.0
ABS_MAX        = 8.0
N_FRAMES       = 150


class OutLayerLoRA(nn.Module):
    def __init__(self, rank=4, in_dim=DEC_OUT_IN_DIM, color_dim=COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def render_mesh(mesh, renderer):
    ext  = EXTRINSICS.to(DEVICE)
    intr = INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    mask = res['mask'].unsqueeze(0)
    return res['color'] * mask + (1.0 - mask)


def colonly_forward(dec_model, lora, slat_norm):
    def _hook(mod, inp, out):
        x_feats   = inp[0].feats
        delta     = lora(x_feats)
        new_feats = out.feats.clone()
        new_feats[:, COLOR_START:COLOR_END] = (
            new_feats[:, COLOR_START:COLOR_END] + delta
        ).clamp(ABS_MIN, ABS_MAX)
        return out.replace(new_feats)
    with torch.no_grad():
        h = dec_model.out_layer.register_forward_hook(_hook)
        meshes = dec_model(slat_norm)
        h.remove()
    return meshes[0]


def make_comparison_strip(gt_path, render_tensor, frozen_render, frame_idx, alpha):
    cell = RENDER_RES
    label_h = 28
    panels = [
        (gt_path,       f'GT  f{frame_idx:04d}  α={alpha:.3f}'),
        (frozen_render, 'frozen decoder'),
        (render_tensor, f'DynaMesh  α={alpha:.3f}'),
    ]
    W = cell * len(panels)
    canvas = Image.new('RGB', (W, cell + label_h), (15, 15, 15))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for col, (src, lbl) in enumerate(panels):
        if isinstance(src, (str, Path)):
            img = Image.open(src).convert('RGB').resize((cell, cell), Image.LANCZOS)
        else:
            arr = (src.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((cell, cell), Image.LANCZOS)
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col * cell, 0, (col + 1) * cell - 1, label_h - 1], fill=(30, 30, 45))
        try:    tw = draw.textbbox((0, 0), lbl)[2]
        except: tw = len(lbl) * 7
        draw.text((col * cell + (cell - tw) // 2, 6), lbl, fill=(210, 210, 210))
    return canvas


def main():
    print(f'Run:  {RUN_DIR}', flush=True)
    print(f'Ckpt: {CKPT}', flush=True)

    # Load SLAT cache  — shape: slats(150,7301,8), coords(7301,4)
    print('[SLAT] Loading cache...', flush=True)
    raw        = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr  = raw['slats']   # (150, 7301, 8)  — already normalized
    coords_arr = raw['coords']  # (7301, 4)  — shared across frames
    slat_cache = {fi: (
        torch.from_numpy(slats_arr[fi - 1].copy()),
        torch.from_numpy(coords_arr.copy()),
    ) for fi in range(1, N_FRAMES + 1)}
    print(f'[SLAT] {len(slat_cache)} frames  feats={slats_arr.shape}', flush=True)

    # Load decoder
    print('[MODEL] Loading TRELLIS decoder...', flush=True)
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe     = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec_model = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec_model.parameters():
        p.requires_grad_(False)

    # Load LoRA
    lora = OutLayerLoRA(rank=4).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    lora.load_state_dict(ckpt['lora_state'])
    lora.eval()
    print(f'[LORA] Loaded epoch={ckpt["epoch"]}  psnr={ckpt["best_psnr"]:.3f}', flush=True)

    # Renderer
    renderer = make_renderer(DEVICE)

    # Precompute frozen renders for comparison (just once per frame)
    frozen_cache = {}

    # Render all frames
    render_frames = []
    strip_frames  = []

    for fi in range(1, N_FRAMES + 1):
        alpha = (fi - 1) / (N_FRAMES - 1)
        feats, coords = slat_cache[fi]
        # cache is already normalized — create SparseTensor directly
        slat_norm = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))

        # DynaMesh render
        mesh   = colonly_forward(dec_model, lora, slat_norm)
        color  = render_mesh(mesh, renderer).detach().clamp(0, 1)

        # Frozen baseline (cached)
        if fi not in frozen_cache:
            with torch.no_grad():
                meshes_base = dec_model(slat_norm)
            frozen_cache[fi] = render_mesh(meshes_base[0], renderer).detach().clamp(0, 1)

        # Save render frame
        arr = (color.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        frame_path = OUT_DIR / f'render_f{fi:04d}.png'
        img.save(frame_path)
        render_frames.append(frame_path)

        # Save comparison strip
        gt_path = GT_FRAMES_DIR / f'frame_{fi:04d}.png'
        strip   = make_comparison_strip(gt_path, color, frozen_cache[fi], fi, alpha)
        strip_path = OUT_DIR / f'strip_f{fi:04d}.png'
        strip.save(strip_path)
        strip_frames.append(strip_path)

        if fi % 25 == 0 or fi == N_FRAMES:
            print(f'  rendered {fi}/{N_FRAMES}  α={alpha:.3f}', flush=True)

        del mesh, color, slat_norm
        gc.collect()
        torch.cuda.empty_cache()

    # Make videos with ffmpeg
    import subprocess

    video_render = str(RUN_DIR / 'dynamic_render.mp4')
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(args.fps),
        '-i', str(OUT_DIR / 'render_f%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        video_render
    ], check=True)
    print(f'[VIDEO] {video_render}', flush=True)

    video_strip = str(RUN_DIR / 'comparison_strip.mp4')
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', str(args.fps),
        '-i', str(OUT_DIR / 'strip_f%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        video_strip
    ], check=True)
    print(f'[VIDEO] {video_strip}', flush=True)

    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
