"""
debug_render_v6_absclamp.py
---------------------------
v6 fix: absolute value clamp on LoRA color channels instead of delta clamp.

Root cause of v4/v5 shadow + white patches:
  delta clamp allows:  frozen_base(+7) + delta(+15) = +22  → sigmoid(22)=1.0=white
  delta clamp blocks:  frozen_base(-8) + delta(+13) capped → result stays dark=shadow

Fix: clamp the final LoRA color values to frozen's natural range [-9, 8].
  sigmoid(8.0)  = 0.9997  → bright orange, not white
  sigmoid(-9.0) = 0.0001  → near-black, same floor as frozen

  mixed_colors = lora_colors.clamp(ABS_MIN, ABS_MAX)

No delta involved → no shadow, no white.

Usage:
  python debug_render_v6_absclamp.py \
    --run-dir experiments/lora_experiments/runs/rung5_54_r4_s6_31e5b3e8 \
    --out-dir experiments/decoder_same_geometry/rendered_videos/rung5_54_v6_absclamp \
    --fps 12 \
    2>&1 | tee experiments/decoder_same_geometry/logs/render_v6_absclamp_$(date +%Y%m%d_%H%M%S).log
"""

import sys, os, json, gc, math, argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))

os.environ['HF_HOME']    = '/net/scratch/rajhansini/.cache/huggingface'
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
PIPELINE_CKPT = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES      = 150
DEVICE        = torch.device('cuda')

# Frozen color range: min=-8.76  max=7.27
# Clamp LoRA to [-9, 8] → sigmoid(8)=0.9997, no white; full expressive range preserved.
ABS_MIN_DEFAULT = -9.0
ABS_MAX_DEFAULT =  8.0

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir',  required=True, type=Path)
ap.add_argument('--out-dir',  required=True, type=Path)
ap.add_argument('--fps',      default=12,    type=int)
ap.add_argument('--abs-min',  default=ABS_MIN_DEFAULT, type=float,
                help=f'Absolute lower clamp on LoRA colors (default {ABS_MIN_DEFAULT})')
ap.add_argument('--abs-max',  default=ABS_MAX_DEFAULT, type=float,
                help=f'Absolute upper clamp on LoRA colors (default {ABS_MAX_DEFAULT})')
args = ap.parse_args()

RUN_DIR = args.run_dir.resolve()
OUT_DIR = args.out_dir.resolve()
ABS_MIN = args.abs_min
ABS_MAX = args.abs_max
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg  = json.loads((RUN_DIR / 'config.json').read_text())
RANK = cfg.get('rank', 4)
print(f'[CONFIG] run={RUN_DIR.name}  rank={RANK}  abs_clamp=[{ABS_MIN}, {ABS_MAX}]')

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS,
)

# ── LoRA classes ──────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class DecLoRABundle(nn.Module):
    def __init__(self, dim=768, rank=4):
        super().__init__()
        mlp_h = dim * 4
        self.lora_qkv = LoRALayer(dim, 3 * dim, rank)
        self.lora_out = LoRALayer(dim, dim,     rank)
        self.lora_fc1 = LoRALayer(dim, mlp_h,  rank)
        self.lora_fc2 = LoRALayer(mlp_h, dim,  rank)


class DecLoRARegistry(nn.Module):
    def __init__(self, active, rank=4):
        super().__init__()
        self.blocks = nn.ModuleDict({str(i): DecLoRABundle(rank=rank) for i in active})

    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


@contextmanager
def dec_lora_ctx(dec_model, registry):
    handles = []
    for i, block in enumerate(dec_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue
        def _qkv(mod, inp, out, _lb=lb):
            return out + _lb.lora_qkv(inp[0]).to(out.dtype)
        def _out_h(mod, inp, out, _lb=lb):
            return out + _lb.lora_out(inp[0]).to(out.dtype)
        def _fc1(mod, inp, out, _lb=lb):
            d = _lb.lora_fc1(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))
        def _fc2(mod, inp, out, _lb=lb):
            d = _lb.lora_fc2(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))
        handles.append(block.attn.to_qkv.register_forward_hook(_qkv))
        handles.append(block.attn.to_out.register_forward_hook(_out_h))
        handles.append(block.mlp.mlp[0].register_forward_hook(_fc1))
        handles.append(block.mlp.mlp[2].register_forward_hook(_fc2))
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ── load pipeline ─────────────────────────────────────────────────────────────
print('\n[LOAD] TRELLIS pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PIPELINE_CKPT)
pipeline.cuda()

flow_model = pipeline.models['slat_flow_model']
dec_model  = pipeline.models['slat_decoder_mesh']
dino_model = pipeline.models['image_cond_model']
for m in [flow_model, dec_model, dino_model]:
    for p in m.parameters():
        p.requires_grad_(False)

renderer = make_renderer()
EXT  = EXTRINSICS.to(DEVICE)
INTR = INTRINSICS.to(DEVICE)

_layouts = dec_model.mesh_extractor.layouts
COLOR_START, COLOR_END = _layouts['color']['range']
print(f'[LAYOUT] color channels: {COLOR_START}:{COLOR_END}')
print(f'[CLAMP]  abs=[{ABS_MIN}, {ABS_MAX}]  '
      f'sigmoid({ABS_MAX})={1/(1+math.exp(-ABS_MAX)):.4f}  '
      f'sigmoid({ABS_MIN})={1/(1+math.exp(-ABS_MIN)):.4f}')

# ── load SLaT cache ───────────────────────────────────────────────────────────
SHARED_CACHE = ROOT / 'experiments' / 'decoder_same_geometry' / 'rung5_slat_cache.npz'
_cache_path  = RUN_DIR / 'slat_cache.npz'
SLAT_CACHE   = _cache_path if _cache_path.exists() else SHARED_CACHE

print(f'[SLAT] Loading from cache: {SLAT_CACHE}')
cache     = np.load(SLAT_CACHE)
slats_arr = cache['slats']
coords_np = cache['coords']
coords_t  = torch.from_numpy(coords_np).int()
print(f'[SLAT] N_vox={slats_arr.shape[1]}  frames={slats_arr.shape[0]}')
slats = {fi: (torch.from_numpy(slats_arr[fi - 1]).float(), coords_t)
         for fi in range(1, N_FRAMES + 1)}

flow_model.cpu(); dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

# ── load LoRA ─────────────────────────────────────────────────────────────────
active   = cfg.get('active_blocks', list(range(12)))
registry = None

if active:
    best            = RUN_DIR / 'lora_ckpts' / 'lora_best.pt'
    ckpt_candidates = sorted(RUN_DIR.glob('lora_ckpts/lora_e*.pt'))
    ckpt_path       = best if best.exists() else (ckpt_candidates[-1] if ckpt_candidates else None)

    if ckpt_path is not None:
        print(f'[CKPT] {ckpt_path.name}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'registry_state' in ckpt:
            registry = DecLoRARegistry(active, rank=RANK)
            registry.load_state_dict(ckpt['registry_state'])
            registry.to(DEVICE).eval()
        else:
            print('[CKPT] No registry_state — rendering frozen only')
    else:
        print('[CKPT] No checkpoint found — rendering frozen only')

dec_model.to(DEVICE)

# ── helpers ───────────────────────────────────────────────────────────────────
def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def decode_both(feats, crds):
    """
    Two-pass out_layer interception.
    Geometry (0:COLOR_START): always from frozen.
    Colors (COLOR_START:COLOR_END): LoRA values clamped to [ABS_MIN, ABS_MAX].

    sigmoid(ABS_MAX=8) = 0.9997 → bright but not white.
    No delta involved → no shadow from tight delta restriction.
    """
    st       = sp.SparseTensor(feats=feats.to(DEVICE), coords=crds.to(DEVICE))
    captured = {}

    def _hook(key):
        def _fn(mod, inp, out):
            captured[key] = out
        return _fn

    with torch.no_grad():
        h1            = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        frozen_meshes = dec_model(st)
        h1.remove()
        frozen_mesh   = filter_degenerate_faces(frozen_meshes[0])

        if registry is None:
            return frozen_mesh, frozen_mesh

        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_lora_ctx(dec_model, registry):
            _ = dec_model(st)
        h2.remove()

    frozen_h    = captured['frozen']
    lora_h      = captured['lora']
    mixed_feats = frozen_h.feats.clone()

    # Absolute clamp: LoRA colors clamped to frozen's natural range.
    lora_col = lora_h.feats[:, COLOR_START:COLOR_END].to(mixed_feats.dtype)
    mixed_feats[:, COLOR_START:COLOR_END] = lora_col.clamp(ABS_MIN, ABS_MAX)

    mixed_h = frozen_h.replace(mixed_feats)
    with torch.no_grad():
        lora_meshes = dec_model.to_representation(mixed_h)
    return frozen_mesh, filter_degenerate_faces(lora_meshes[0])


def render_mesh(mesh):
    res  = renderer.render(mesh, EXT, INTR, return_types=['color', 'mask'])
    mask = res['mask'].unsqueeze(0)
    col  = res['color'] * mask + (1.0 - mask)
    return (col.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def load_gt(fi):
    return np.array(
        Image.open(GT_FRAMES_DIR / f'frame_{fi:04d}.png')
             .convert('RGB')
             .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    )

# ── render loop ───────────────────────────────────────────────────────────────
frames_dir = OUT_DIR / 'frames'
frames_dir.mkdir(exist_ok=True)

LABEL_H  = 24
LABEL_BG = (15, 15, 15)

def labeled_strip(arrays, labels, colors):
    W = arrays[0].shape[1]; H = arrays[0].shape[0]
    canvas = np.zeros((H + LABEL_H, W * len(arrays), 3), dtype=np.uint8)
    canvas[:] = LABEL_BG
    pil  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, (arr, lbl, col) in enumerate(zip(arrays, labels, colors)):
        pil.paste(Image.fromarray(arr), (i * W, LABEL_H))
        draw.text((i * W + 6, 4), lbl, fill=col)
    return pil

run_label = RUN_DIR.name[:38]
print(f'\n[RENDER] {N_FRAMES} frames → {frames_dir}')
for fi in range(1, N_FRAMES + 1):
    ff, fc = slats[fi]
    gt_arr     = load_gt(fi)
    frozen_mesh, lora_mesh = decode_both(ff, fc)
    frozen_arr = render_mesh(frozen_mesh)
    lora_arr   = render_mesh(lora_mesh)

    strip = labeled_strip(
        [gt_arr, frozen_arr, lora_arr],
        ['GT video', 'Frozen baseline', run_label],
        [(80, 200, 80), (200, 80, 80), (80, 160, 255)],
    )
    strip.save(frames_dir / f'frame_{fi:04d}.png')
    if fi % 30 == 0 or fi == 1:
        print(f'  frame {fi}/{N_FRAMES}')

dec_model.cpu()
print('[RENDER] Done.')

# ── ffmpeg video ──────────────────────────────────────────────────────────────
out_video  = OUT_DIR / f'{RUN_DIR.name}_v6_absclamp.mp4'
_ffmpeg    = '/usr/bin/ffmpeg'
_enc_probe = subprocess.run([_ffmpeg, '-encoders'], capture_output=True, text=True)
if 'libx264' in _enc_probe.stdout:
    _codec_flags = ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
elif 'libopenh264' in _enc_probe.stdout:
    _codec_flags = ['-c:v', 'libopenh264', '-b:v', '4M', '-pix_fmt', 'yuv420p']
elif 'h264_nvenc' in _enc_probe.stdout:
    _codec_flags = ['-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '18', '-pix_fmt', 'yuv420p']
else:
    _codec_flags = ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p']

print(f'[FFMPEG] encoder: {_codec_flags[1]}')
cmd = [
    _ffmpeg, '-y',
    '-framerate', str(args.fps),
    '-pattern_type', 'glob',
    '-i', str(frames_dir / 'frame_*.png'),
    *_codec_flags,
    str(out_video),
]
subprocess.run(cmd, check=True)
print(f'[VIDEO]  {out_video}')
