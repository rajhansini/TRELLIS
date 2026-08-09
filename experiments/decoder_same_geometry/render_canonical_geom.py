"""
render_canonical_geom.py
------------------------
Renders 150 frames with CANONICAL geometry (from frame 75, frozen decoder)
but PER-FRAME color from the LoRA decoder.

Fix for inconsistent triangulation: the out_layer output is intercepted and
SDF+deform+weights channels are replaced with canonical values, while the
color channels (53:101) come from the LoRA per frame.

Usage:
  python render_canonical_geom.py --run-dir runs/rung5_54_r4_s6_31e5b3e8 \
                                   --out-dir /tmp/out/rung5_54 --fps 12
"""

import sys, os, json, gc, math, argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch.nn.functional as F
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
CANONICAL_FRAME = 75
STRUCT_SEED = 42
NOISE_SEED  = 6
N_FRAMES    = 150
DEVICE      = torch.device('cuda')
STEPS       = 25
RESCALE_T   = 3.0

_t_seq = np.linspace(1, 0, STEPS + 1)
_t_seq = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True, type=Path)
ap.add_argument('--out-dir', required=True, type=Path)
ap.add_argument('--fps',     default=12, type=int)
args = ap.parse_args()

RUN_DIR = args.run_dir.resolve()
OUT_DIR = args.out_dir.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg  = json.loads((RUN_DIR / 'config.json').read_text())
RANK = cfg.get('rank', 4)
print(f'[CONFIG] run={RUN_DIR.name}  rank={RANK}')

# ── imports after env setup ───────────────────────────────────────────────────
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)
import torchvision.transforms as T

# ── LoRA classes (decoder only, dim=768) ─────────────────────────────────────
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
EXT  = EXTRINSICS.to(DEVICE)
INTR = INTRINSICS.to(DEVICE)

# Color channel range from the mesh extractor layout
_layouts = dec_model.mesh_extractor.layouts
COLOR_START, COLOR_END = _layouts['color']['range']
print(f'[LAYOUT] color channels: {COLOR_START}:{COLOR_END}  (total feats_channels={dec_model.mesh_extractor.feats_channels})')

# ── load SLaTs: per-run cache → shared cache → flow model for all 150 frames ─
SHARED_CACHE = ROOT / 'experiments' / 'decoder_same_geometry' / 'rung5_slat_cache.npz'
_cache_path  = RUN_DIR / 'slat_cache.npz'
SLAT_CACHE   = _cache_path if _cache_path.exists() else SHARED_CACHE
slats = {}

if SLAT_CACHE.exists():
    print(f'[SLAT] Loading from cache: {SLAT_CACHE}')
    cache     = np.load(SLAT_CACHE)
    slats_arr = cache['slats']    # (150, N_vox, 8)
    coords_np = cache['coords']   # (N_vox, 4)  — includes batch dim
    coords_t  = torch.from_numpy(coords_np).long()
    N_vox     = slats_arr.shape[1]
    print(f'[SLAT] N_vox={N_vox}  frames={slats_arr.shape[0]}')
    for fi in range(1, N_FRAMES + 1):
        ff = torch.from_numpy(slats_arr[fi - 1]).float()
        slats[fi] = (ff, coords_t)
    flow_model.cpu(); dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    print('[SLAT] Done (from cache).')
else:
    print('[SLAT] No cache found — encoding DINOv2 + running flow model...')
    _dino_tfm = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def encode_frame(fi):
        img = Image.open(GT_FRAMES_DIR / f'frame_{fi:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
        t   = _dino_tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = dino_model(t, is_training=True)['x_prenorm']
            toks  = F.layer_norm(feats, feats.shape[-1:])
        return toks.squeeze(0).cpu()

    print('[DINO] Encoding all 150 frames...')
    raw_tokens = {}
    for fi in range(1, N_FRAMES + 1):
        raw_tokens[fi] = encode_frame(fi)
        if fi % 30 == 0:
            print(f'  {fi}/{N_FRAMES}')

    print('[STRUCT] Sampling structure...')
    struct_img  = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([pipeline.preprocess_image(struct_img)])
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(STRUCT_SEED)
    with torch.no_grad():
        coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1).to(DEVICE)
    N_vox = coords.shape[0]
    print(f'[STRUCT] N_vox={N_vox}')

    torch.manual_seed(NOISE_SEED)
    fixed_noise = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    def full_denoise(cond_gl):
        ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                v = flow_model(ns, t_ten, cond_gl)
                ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
        return normalize_slat(ns)

    print('[SLAT] Denoising 150 frames (frozen flow)...')
    for fi in range(1, N_FRAMES + 1):
        cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        s = full_denoise(cond)
        slats[fi] = (s.feats.cpu(), s.coords.cpu())
        if fi % 30 == 0:
            print(f'  {fi}/{N_FRAMES}')

    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    print('[SLAT] Done.')

# ── load LoRA ─────────────────────────────────────────────────────────────────
active = cfg.get('active_blocks', list(range(12)))
registry = None

if active:
    best = RUN_DIR / 'lora_ckpts' / 'lora_best.pt'
    ckpt_candidates = sorted(RUN_DIR.glob('lora_ckpts/lora_e*.pt'))
    ckpt_path = best if best.exists() else (ckpt_candidates[-1] if ckpt_candidates else None)

    if ckpt_path is not None:
        print(f'[CKPT] {ckpt_path.name}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'registry_state' in ckpt:
            registry = DecLoRARegistry(active, rank=RANK)
            registry.load_state_dict(ckpt['registry_state'])
            registry.to(DEVICE).eval()
        else:
            print(f'[CKPT] No registry_state found (keys: {list(ckpt.keys())}) — rendering frozen only')
    else:
        print('[CKPT] No checkpoint found — rendering frozen only')
else:
    print('[CKPT] No active blocks (frozen baseline) — rendering frozen only')

dec_model.to(DEVICE)

# ── filter degenerate faces ────────────────────────────────────────────────────
def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh

# ── decode + render helpers ────────────────────────────────────────────────────
def decode_lora(feats, crds):
    """LoRA decoder, direct decode. Falls back to frozen if no registry."""
    if registry is None:
        return decode_frozen(feats, crds)
    st = sp.SparseTensor(feats=feats.to(DEVICE), coords=crds.to(DEVICE))
    with torch.no_grad():
        with dec_lora_ctx(dec_model, registry):
            mesh = dec_model(st)[0]
    return filter_degenerate_faces(mesh)


def decode_frozen(feats, crds):
    """Frozen decoder, no LoRA, no geometry swap."""
    st = sp.SparseTensor(feats=feats.to(DEVICE), coords=crds.to(DEVICE))
    with torch.no_grad():
        mesh = dec_model(st)[0]
    return filter_degenerate_faces(mesh)


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

# ── render loop ────────────────────────────────────────────────────────────────
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
    frozen_arr = render_mesh(decode_frozen(ff, fc))
    lora_arr   = render_mesh(decode_lora(ff, fc))

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

# ── ffmpeg video ───────────────────────────────────────────────────────────────
out_video = OUT_DIR / f'{RUN_DIR.name}_canonical_geom.mp4'
cmd = [
    'ffmpeg', '-y',
    '-framerate', str(args.fps),
    '-pattern_type', 'glob',
    '-i', str(frames_dir / 'frame_*.png'),
    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
    str(out_video),
]
print(f'\n[FFMPEG] Encoding...')
subprocess.run(cmd, check=True)
print(f'[VIDEO]  {out_video}')
