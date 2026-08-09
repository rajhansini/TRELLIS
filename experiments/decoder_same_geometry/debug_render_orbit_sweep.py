"""
debug_render_orbit_sweep.py
---------------------------
Renders a LoRA-textured mesh from N_ANGLES camera positions (azimuth sweep)
to produce a turntable orbit video.

Decodes a SINGLE SLaT frame (--frame, default 75) once, then renders frozen
and LoRA meshes from every angle. Uses the same two-pass out_layer interception
+ clamped-delta approach as debug_render_clamped_colors.py.

Extrinsics orbit formula (derived from confirmed front-view EXTRINSICS):
  world_up = (0, 0, -1)  [world -Z is image top, matches original cam_y]
  theta=0  → original front view  (camera at world (0,-r,0))
  theta=90 → camera at world (+r, 0, 0)  [right side]

Usage:
  python debug_render_orbit_sweep.py \
    --run-dir experiments/lora_experiments/runs/rung5_54_r4_s6_31e5b3e8 \
    --out-dir experiments/decoder_same_geometry/orbit_videos/rung5_54_orbit \
    --frame 75 --delta-clip 15 --n-angles 30 --elevation 15 --fps 12 \
    2>&1 | tee experiments/decoder_same_geometry/logs/orbit_rung5_54_$(date +%Y%m%d_%H%M%S).log
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

PIPELINE_CKPT = 'JeffreyXiang/TRELLIS-image-large'
DEVICE        = torch.device('cuda')
ORBIT_RADIUS  = 2.0

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir',    required=True, type=Path)
ap.add_argument('--out-dir',    required=True, type=Path)
ap.add_argument('--frame',      default=75,    type=int,
                help='Which SLaT frame to decode (1-150, default 75)')
ap.add_argument('--delta-clip', default=15.0,  type=float,
                help='Max |LoRA color delta| before sigmoid (default 15.0)')
ap.add_argument('--n-angles',   default=30,    type=int,
                help='Number of orbit angles (default 30)')
ap.add_argument('--elevation',  default=0.0,   type=float,
                help='Camera elevation in degrees above equator (default 15)')
ap.add_argument('--fps',        default=12,    type=int)
args = ap.parse_args()

RUN_DIR    = args.run_dir.resolve()
OUT_DIR    = args.out_dir.resolve()
DELTA_CLIP = args.delta_clip
N_ANGLES   = args.n_angles
ELEVATION  = args.elevation
FRAME      = args.frame
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg  = json.loads((RUN_DIR / 'config.json').read_text())
RANK = cfg.get('rank', 4)
print(f'[CONFIG] run={RUN_DIR.name}  rank={RANK}  frame={FRAME}  '
      f'delta_clip={DELTA_CLIP}  n_angles={N_ANGLES}  elevation={ELEVATION}°')

# ── imports after env setup ───────────────────────────────────────────────────
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, INTRINSICS,
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
pipeline = TrellisImageTo3DPipeline.from_pretrained(PIPELINE_CKPT)
pipeline.cuda()

flow_model = pipeline.models['slat_flow_model']
dec_model  = pipeline.models['slat_decoder_mesh']
dino_model = pipeline.models['image_cond_model']
for m in [flow_model, dec_model, dino_model]:
    for p in m.parameters():
        p.requires_grad_(False)

renderer = make_renderer()
INTR     = INTRINSICS.to(DEVICE)

_layouts    = dec_model.mesh_extractor.layouts
COLOR_START, COLOR_END = _layouts['color']['range']
print(f'[LAYOUT] color channels: {COLOR_START}:{COLOR_END}  '
      f'total={dec_model.mesh_extractor.feats_channels}')
print(f'[CLAMP]  delta_clip={DELTA_CLIP}')

# ── load SLaT cache ───────────────────────────────────────────────────────────
SHARED_CACHE = ROOT / 'experiments' / 'decoder_same_geometry' / 'rung5_slat_cache.npz'
_cache_path  = RUN_DIR / 'slat_cache.npz'
SLAT_CACHE   = _cache_path if _cache_path.exists() else SHARED_CACHE

print(f'[SLAT] Loading from cache: {SLAT_CACHE}')
cache     = np.load(SLAT_CACHE)
slats_arr = cache['slats']
coords_np = cache['coords']
coords_t  = torch.from_numpy(coords_np).int()
N_vox     = slats_arr.shape[1]
print(f'[SLAT] N_vox={N_vox}  total_frames={slats_arr.shape[0]}  using frame={FRAME}')

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

# ── orbit extrinsics ──────────────────────────────────────────────────────────
def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=ORBIT_RADIUS):
    """
    World-to-camera extrinsics for orbit at azimuth theta, elevation el.

    theta=0  → original front view (camera at world (0, -radius, 0))
    world_up = (0,0,-1) matches original cam_y, keeping teapot upright across orbit.
    """
    theta = math.radians(theta_deg)
    el    = math.radians(elevation_deg)

    cx = radius * math.sin(theta) * math.cos(el)
    cy = -radius * math.cos(theta) * math.cos(el)
    cz = radius * math.sin(el)
    C  = torch.tensor([cx, cy, cz], dtype=torch.float32)

    world_up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cam_z    = -C / C.norm()
    cam_x    = torch.linalg.cross(world_up, cam_z)
    cam_x    = cam_x / cam_x.norm()
    cam_y    = torch.linalg.cross(cam_z, cam_x)
    cam_y    = cam_y / cam_y.norm()

    R = torch.stack([cam_x, cam_y, cam_z], dim=0)  # (3,3)
    t = R @ (-C)

    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3,  3] = t
    return E.to(DEVICE)

# ── helpers ───────────────────────────────────────────────────────────────────
def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def decode_both(feats, crds):
    """Two-pass out_layer interception: frozen geometry + clamped LoRA colors."""
    st       = sp.SparseTensor(feats=feats.to(DEVICE), coords=crds.to(DEVICE))
    captured = {}

    def _hook(key):
        def _fn(mod, inp, out):
            captured[key] = out
        return _fn

    with torch.no_grad():
        h1             = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        frozen_meshes  = dec_model(st)
        h1.remove()
        frozen_mesh    = filter_degenerate_faces(frozen_meshes[0])

        if registry is None:
            return frozen_mesh, frozen_mesh

        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_lora_ctx(dec_model, registry):
            _ = dec_model(st)
        h2.remove()

    frozen_h = captured['frozen']
    lora_h   = captured['lora']

    mixed_feats = frozen_h.feats.clone()
    fz_col      = frozen_h.feats[:, COLOR_START:COLOR_END]
    lora_col    = lora_h.feats[:, COLOR_START:COLOR_END].to(fz_col.dtype)
    delta       = (lora_col - fz_col).clamp(-DELTA_CLIP, DELTA_CLIP)
    mixed_feats[:, COLOR_START:COLOR_END] = fz_col + delta

    mixed_h = frozen_h.replace(mixed_feats)
    with torch.no_grad():
        lora_meshes = dec_model.to_representation(mixed_h)
    lora_mesh = filter_degenerate_faces(lora_meshes[0])

    return frozen_mesh, lora_mesh


def render_mesh(mesh, ext):
    res  = renderer.render(mesh, ext, INTR, return_types=['color', 'mask'])
    mask = res['mask'].unsqueeze(0)
    col  = res['color'] * mask + (1.0 - mask)
    return (col.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# ── decode once ───────────────────────────────────────────────────────────────
print(f'\n[DECODE] Decoding frame {FRAME}...')
fi         = FRAME
ff         = torch.from_numpy(slats_arr[fi - 1]).float()
frozen_mesh, lora_mesh = decode_both(ff, coords_t)
print(f'[DECODE] Done. frozen faces={frozen_mesh.faces.shape[0]}  '
      f'lora faces={lora_mesh.faces.shape[0]}')

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

angles   = [360.0 * i / N_ANGLES for i in range(N_ANGLES)]
run_label = RUN_DIR.name[:38]

print(f'[RENDER] {N_ANGLES} angles (el={ELEVATION}°) → {frames_dir}')
for idx, theta in enumerate(angles):
    ext        = orbit_extrinsics(theta, elevation_deg=ELEVATION)
    frozen_arr = render_mesh(frozen_mesh, ext)
    lora_arr   = render_mesh(lora_mesh,   ext)

    angle_label = f'{theta:.0f}°'
    strip = labeled_strip(
        [frozen_arr, lora_arr],
        [f'Frozen  {angle_label}', f'{run_label}  {angle_label}'],
        [(200, 80, 80), (80, 160, 255)],
    )
    strip.save(frames_dir / f'frame_{idx + 1:04d}.png')
    if idx % 10 == 0 or idx == N_ANGLES - 1:
        print(f'  angle {idx + 1}/{N_ANGLES}  θ={theta:.0f}°')

dec_model.cpu()
print('[RENDER] Done.')

# ── ffmpeg video ──────────────────────────────────────────────────────────────
out_video  = OUT_DIR / f'{RUN_DIR.name}_orbit.mp4'
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
