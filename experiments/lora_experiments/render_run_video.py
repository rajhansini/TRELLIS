"""
render_run_video.py
-------------------
Given a run directory, renders all 150 frames side-by-side:
    GT  |  Frozen baseline  |  LoRA prediction

Auto-detects rung type from config.json:
  stage key       → rung5 (decoder LoRA, registry_state, DEC_DIM=768)
  layer_set key   → rung3 (flow LoRA, 5 layers/block, lora_state)
  blocks key only → rung2 (flow LoRA, 2 layers/block, lora_state)
  run key         → rung1 (flow LoRA, all 24 blocks, lora_state with "0." prefix)

Usage:
  python render_run_video.py --run-dir runs/rung5_54_r4_s6_31e5b3e8 \
                             --out-dir /tmp/render_out/rung5_54 --fps 12
"""

import sys, os, json, gc, math, argparse, subprocess
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path('/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))

os.environ['HF_HOME']    = '/net/scratch/rajhansini/.cache/huggingface'
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

GT_FRAMES_DIR  = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
PIPELINE_CKPT  = 'JeffreyXiang/TRELLIS-image-large'
STRUCT_SEED    = 42
NOISE_SEED     = 6
N_FRAMES       = 150
DEVICE         = torch.device('cuda')
STEPS          = 25
RESCALE_T      = 3.0

_t_seq = np.linspace(1, 0, STEPS + 1)
_t_seq = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

# ── args ──────────────────────────────────────────────────────────────────────
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

if 'stage' in cfg:
    RUNG = 'rung5'
elif 'layer_set' in cfg:
    RUNG = 'rung3'
elif 'blocks' in cfg:
    RUNG = 'rung2'
else:
    RUNG = 'rung1'

print(f'[CONFIG] run={RUN_DIR.name}  rung={RUNG}  rank={RANK}')

# ── nvdiffrast build guard ────────────────────────────────────────────────────
def _ensure_nvdiffrast():
    try:
        import nvdiffrast.torch as dr
        dr.RasterizeCudaContext()
        return
    except Exception:
        pass
    arch = subprocess.check_output(
        [sys.executable, '-c',
         'import torch; c=torch.cuda.get_device_capability(); print(f"{c[0]}_{c[1]}")'],
        text=True).strip()
    local = f'/tmp/nvdiffrast_{arch}'
    src   = local + '_src'
    Path(src).mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           'CUDA_HOME': '/usr/local/cuda',
           'PATH': f'/usr/local/cuda/bin:{os.environ["PATH"]}',
           'LD_LIBRARY_PATH': f'/usr/local/cuda/lib64:{os.environ.get("LD_LIBRARY_PATH","")}'}
    if not (Path(src) / 'nvdiffrast').exists():
        subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                        f'{src}/nvdiffrast', '--depth', '1', '--quiet'],
                       check=True, env=env)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--target', local, f'{src}/nvdiffrast'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    sys.path.insert(0, local)

_ensure_nvdiffrast()

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)
import torchvision.transforms as T

# ── LoRA class definitions ────────────────────────────────────────────────────
# Flow model: dim=1024, mlp_h=4096
# Decoder:    dim=768,  mlp_h=3072

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


# rung1 / rung2: Q + KV in flow model cross-attn (dim=1024)
class FlowLoRABlock2(nn.Module):
    def __init__(self, dim=1024, rank=4):
        super().__init__()
        self.lora_q  = LoRALayer(dim, dim,     rank)
        self.lora_kv = LoRALayer(dim, 2 * dim, rank)

class FlowLoRARegistry2(nn.Module):
    def __init__(self, active, dim=1024, rank=4):
        super().__init__()
        self.blocks = nn.ModuleDict({str(i): FlowLoRABlock2(dim, rank) for i in active})

    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


# rung3: Q + KV + out + fc1 + fc2 in flow model (dim=1024)
class FlowLoRABlock5(nn.Module):
    def __init__(self, dim=1024, rank=4):
        super().__init__()
        self.lora_q   = LoRALayer(dim,     dim,     rank)
        self.lora_kv  = LoRALayer(dim,     2 * dim, rank)
        self.lora_out = LoRALayer(dim,     dim,     rank)
        self.lora_fc1 = LoRALayer(dim,     4 * dim, rank)
        self.lora_fc2 = LoRALayer(4 * dim, dim,     rank)

class FlowLoRARegistry5(nn.Module):
    def __init__(self, active, dim=1024, rank=4):
        super().__init__()
        self.blocks = nn.ModuleDict({str(i): FlowLoRABlock5(dim, rank) for i in active})

    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


# rung5: decoder LoRA — QKV + out + fc1 + fc2 (dim=768)
class DecLoRABundle(nn.Module):
    def __init__(self, dim=768, rank=4):
        super().__init__()
        mlp_h = dim * 4  # 3072
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


# ── LoRA context managers ─────────────────────────────────────────────────────

_ATTN_CHUNK = 256

def _chunked_attn(q, k, v, scale):
    outs = []
    for s in range(0, q.shape[0], _ATTN_CHUNK):
        e  = min(s + _ATTN_CHUNK, q.shape[0])
        qc = q[s:e].float()
        A  = torch.einsum('nhd,mhd->nhm', qc, k.float()) * scale
        A  = torch.softmax(A, -1)
        outs.append(torch.einsum('nhm,mhd->nhd', A, v.float()))
    return torch.cat(outs, 0)


def _flow_fwd2(module, x, context, lb):
    """rung1/rung2: patch cross_attn forward (to_q + to_kv)."""
    ch = module.channels; H = module.num_heads; hd = ch // H
    wdt = module.to_q.weight.dtype
    xf  = x.feats.to(wdt)
    ctx = (context.squeeze(0) if context.dim() == 3 else context).to(wdt)
    with torch.no_grad():
        q_b  = module.to_q(xf)
        kv_b = module.to_kv(ctx)
    if lb is not None:
        q_b  = q_b  + lb.lora_q(xf)
        kv_b = kv_b + lb.lora_kv(ctx)
    k, v = kv_b.reshape(-1, 2, H, hd)[:, 0], kv_b.reshape(-1, 2, H, hd)[:, 1]
    z = _chunked_attn(q_b.reshape(-1, H, hd), k, v, hd ** -0.5).to(wdt).reshape(-1, ch)
    return x.replace(module.to_out(z).to(x.feats.dtype))


def _flow_fwd5(module, x, context, lb):
    """rung3: patch cross_attn + mlp forward (all 5 projection layers)."""
    ch = module.channels; H = module.num_heads; hd = ch // H
    wdt = module.to_q.weight.dtype
    xf  = x.feats.to(wdt)
    ctx = (context.squeeze(0) if context.dim() == 3 else context).to(wdt)
    with torch.no_grad():
        q_b  = module.to_q(xf)
        kv_b = module.to_kv(ctx)
    if lb is not None:
        q_b  = q_b  + lb.lora_q(xf)
        kv_b = kv_b + lb.lora_kv(ctx)
    k, v = kv_b.reshape(-1, 2, H, hd)[:, 0], kv_b.reshape(-1, 2, H, hd)[:, 1]
    z = _chunked_attn(q_b.reshape(-1, H, hd), k, v, hd ** -0.5).to(wdt).reshape(-1, ch)
    with torch.no_grad():
        out = module.to_out(z)
    if lb is not None:
        out = out + lb.lora_out(z)
    return x.replace(out.to(x.feats.dtype))


def _mlp_fwd5(mlp_mod, x, lb):
    """rung3: patch MLP (fc1 + fc2) with LoRA deltas."""
    wdt1 = mlp_mod.mlp[0].weight.dtype
    wdt2 = mlp_mod.mlp[2].weight.dtype
    with torch.no_grad():
        h = mlp_mod.mlp[0](x)
    h = h.replace(h.feats + lb.lora_fc1(x.feats.to(wdt1)).to(h.feats.dtype))
    h = mlp_mod.mlp[1](h)
    with torch.no_grad():
        out = mlp_mod.mlp[2](h)
    return out.replace(out.feats + lb.lora_fc2(h.feats.to(wdt2)).to(out.feats.dtype))


@contextmanager
def flow_lora_ctx(flow_model, reg, five_layers=False):
    saved_ca = {}; saved_mlp = {}; ca_idx = 0
    for blk in flow_model.blocks:
        if not hasattr(blk, 'cross_attn'):
            continue
        ca = blk.cross_attn
        saved_ca[ca_idx] = ca.forward
        lb  = reg.get(ca_idx)
        fwd = _flow_fwd5 if five_layers else _flow_fwd2

        def _make_ca(mod, lb_, fwd_):
            def _f(x, context=None): return fwd_(mod, x, context, lb_)
            return _f
        ca.forward = _make_ca(ca, lb, fwd)

        if five_layers and lb is not None:
            saved_mlp[ca_idx] = blk.mlp.forward
            def _make_mlp(m, lb_):
                def _f(x): return _mlp_fwd5(m, x, lb_)
                return _f
            blk.mlp.forward = _make_mlp(blk.mlp, lb)
        ca_idx += 1
    try:
        yield
    finally:
        ca_idx = 0
        for blk in flow_model.blocks:
            if not hasattr(blk, 'cross_attn'):
                continue
            if ca_idx in saved_ca:
                blk.cross_attn.forward = saved_ca[ca_idx]
            if ca_idx in saved_mlp:
                blk.mlp.forward = saved_mlp[ca_idx]
            ca_idx += 1


@contextmanager
def dec_lora_ctx(dec_model, reg: DecLoRARegistry):
    """rung5: forward hooks on block.attn.to_qkv/to_out and block.mlp.mlp[0/2]."""
    handles = []
    for i, block in enumerate(dec_model.blocks):
        lb = reg.get(i)
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


# ── LoRA build + load ─────────────────────────────────────────────────────────

def find_best_ckpt(run_dir):
    best = run_dir / 'lora_ckpts' / 'lora_best.pt'
    if best.exists():
        return best
    return sorted(run_dir.glob('lora_ckpts/lora_e*.pt'))[-1]


def build_and_load_lora():
    ckpt_path = find_best_ckpt(RUN_DIR)
    print(f'[CKPT]   {ckpt_path.name}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if RUNG == 'rung5':
        active = cfg.get('active_blocks', list(range(12)))
        reg = DecLoRARegistry(active, rank=RANK)
        reg.load_state_dict(ckpt['registry_state'])
        reg.to(DEVICE).eval()
        return 'dec', reg

    elif RUNG == 'rung3':
        blocks = cfg['blocks']
        reg = FlowLoRARegistry5(blocks, rank=RANK)
        reg.load_state_dict(ckpt['lora_state'])
        reg.to(DEVICE).eval()
        return 'flow5', reg

    elif RUNG == 'rung2':
        blocks = cfg['blocks']
        reg = FlowLoRARegistry2(blocks, rank=RANK)
        reg.load_state_dict(ckpt['lora_state'])
        reg.to(DEVICE).eval()
        return 'flow2', reg

    else:  # rung1: keys like "0.lora_q.A" → rename to "blocks.0.lora_q.A"
        active = list(range(24))
        reg = FlowLoRARegistry2(active, rank=RANK)
        raw = ckpt['lora_state']
        reg.load_state_dict({f'blocks.{k}': v for k, v in raw.items()})
        reg.to(DEVICE).eval()
        return 'flow2', reg


# ── pipeline ──────────────────────────────────────────────────────────────────
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

# ── DINOv2 encode all frames ──────────────────────────────────────────────────
_dino_tfm = T.Compose([
    T.Resize((518, 518), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def encode_frame(fi):
    img = Image.open(GT_FRAMES_DIR / f'frame_{fi:04d}.png').convert('RGB')
    t   = _dino_tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(t)
    return (feats.squeeze(0) if feats.dim() == 3 else feats).cpu()

print('[DINO] Encoding all 150 frames...')
raw_tokens = {}
for fi in range(1, N_FRAMES + 1):
    raw_tokens[fi] = encode_frame(fi)
    if fi % 30 == 0:
        print(f'  {fi}/{N_FRAMES}')

# ── sparse structure (get_cond must run before dino_model.cpu()) ──────────────
print('[STRUCT] Sampling structure...')
struct_img  = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([pipeline.preprocess_image(struct_img)])

dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()
torch.manual_seed(STRUCT_SEED)
with torch.no_grad():
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1).to(DEVICE)
N_vox = coords.shape[0]

print(f'[NOISE] seed={NOISE_SEED}  N_vox={N_vox}')
torch.manual_seed(NOISE_SEED)
fixed_noise = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

# ── flow model denoise ────────────────────────────────────────────────────────

def full_denoise(cond_gl, lora_type=None, lora_reg=None):
    ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            if lora_type in ('flow2', 'flow5'):
                with flow_lora_ctx(flow_model, lora_reg, five_layers=(lora_type == 'flow5')):
                    v = flow_model(ns, t_ten, cond_gl)
            else:
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)

# ── build LoRA and precompute SLaTs ──────────────────────────────────────────
lora_type, lora_reg = build_and_load_lora()
flow_model.to(DEVICE)

print('\n[SLAT] Precomputing 150 SLaTs (frozen + LoRA)...')
slat_frozen = {}
slat_lora   = {}
for fi in range(1, N_FRAMES + 1):
    cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
    sf   = full_denoise(cond, None, None)
    slat_frozen[fi] = (sf.feats.cpu(), sf.coords.cpu())
    if lora_type in ('flow2', 'flow5'):
        sl = full_denoise(cond, lora_type, lora_reg)
        slat_lora[fi] = (sl.feats.cpu(), sl.coords.cpu())
    else:
        slat_lora[fi] = slat_frozen[fi]
    if fi % 30 == 0:
        print(f'  {fi}/{N_FRAMES}')

flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
print('[SLAT] Done.')

# ── render helpers ────────────────────────────────────────────────────────────

def filter_degenerate_faces(mesh, min_area=1e-6):
    """Remove near-zero-area triangle slivers that cause ghost artifacts."""
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    keep = area > min_area
    mesh.faces = f[keep]
    return mesh

def decode_and_render(feats, crds, ctx_fn=None):
    st = sp.SparseTensor(feats=feats.to(DEVICE), coords=crds.to(DEVICE))
    with torch.no_grad():
        if ctx_fn is not None:
            with ctx_fn():
                mesh = dec_model(st)[0]
        else:
            mesh = dec_model(st)[0]
    mesh = filter_degenerate_faces(mesh)
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

LABEL_H  = 24
LABEL_BG = (15, 15, 15)

def labeled_strip(arrays, labels, colors):
    W = arrays[0].shape[1]; H = arrays[0].shape[0]
    total_w = W * len(arrays)
    canvas = np.zeros((H + LABEL_H, total_w, 3), dtype=np.uint8)
    canvas[:] = LABEL_BG
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, (arr, lbl, col) in enumerate(zip(arrays, labels, colors)):
        pil.paste(Image.fromarray(arr), (i * W, LABEL_H))
        draw.text((i * W + 6, 4), lbl, fill=col)
    return pil

# ── render loop ───────────────────────────────────────────────────────────────
dec_model.to(DEVICE)
frames_dir = OUT_DIR / 'frames'
frames_dir.mkdir(exist_ok=True)

lora_ctx = None
if lora_type == 'dec':
    def lora_ctx(): return dec_lora_ctx(dec_model, lora_reg)

run_label = RUN_DIR.name[:38]
print(f'\n[RENDER] Writing {N_FRAMES} frames to {frames_dir}')
for fi in range(1, N_FRAMES + 1):
    gt_arr     = load_gt(fi)
    ff, fc     = slat_frozen[fi]
    frozen_arr = decode_and_render(ff, fc, None)
    lf, lc     = slat_lora[fi]
    lora_arr   = decode_and_render(lf, lc, lora_ctx)

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

# ── make video with ffmpeg ────────────────────────────────────────────────────
out_video = OUT_DIR / f'{RUN_DIR.name}_comparison.mp4'
cmd = [
    'ffmpeg', '-y',
    '-framerate', str(args.fps),
    '-pattern_type', 'glob',
    '-i', str(frames_dir / 'frame_*.png'),
    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
    str(out_video),
]
print(f'\n[FFMPEG] encoding video...')
subprocess.run(cmd, check=True)
print(f'[VIDEO]  {out_video}')
