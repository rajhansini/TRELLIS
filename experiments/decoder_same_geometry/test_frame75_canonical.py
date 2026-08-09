"""
Quick sanity: frame 75 only, 3-way comparison.
Runs flow model for frame 75 only (1 denoise pass) with correct DINOv2 encoding.
~3-4 min total.
"""
import sys, os, json, gc, math
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path('/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))
os.environ['HF_HOME']         = '/net/scratch/rajhansini/.cache/huggingface'
os.environ.setdefault('SPCONV_ALGO',  'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
RUN_DIR       = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs/rung5_53_late_r4_s6_1f298d4d')
OUT_IMG       = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/decoder_same_geometry/frame075_canonical_test.png')
FRAME_IDX     = 75
STRUCT_SEED   = 42
NOISE_SEED    = 6
DEVICE        = torch.device('cuda')
STEPS         = 25
RESCALE_T     = 3.0

_t_seq = np.linspace(1, 0, STEPS + 1)
_t_seq = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)
import torchvision.transforms as T

# ── LoRA ──────────────────────────────────────────────────────────────────────
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
        self.lora_out = LoRALayer(dim, dim, rank)
        self.lora_fc1 = LoRALayer(dim, mlp_h, rank)
        self.lora_fc2 = LoRALayer(mlp_h, dim, rank)

class DecLoRARegistry(nn.Module):
    def __init__(self, active, rank=4):
        super().__init__()
        self.blocks = nn.ModuleDict({str(i): DecLoRABundle(rank=rank) for i in active})
    def get(self, i):
        k = str(i); return self.blocks[k] if k in self.blocks else None

@contextmanager
def dec_lora_ctx(dec_model, registry):
    handles = []
    for i, block in enumerate(dec_model.blocks):
        lb = registry.get(i)
        if lb is None: continue
        def _qkv(mod, inp, out, _lb=lb): return out + _lb.lora_qkv(inp[0]).to(out.dtype)
        def _out_h(mod, inp, out, _lb=lb): return out + _lb.lora_out(inp[0]).to(out.dtype)
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
    try: yield
    finally:
        for h in handles: h.remove()

# ── load pipeline ─────────────────────────────────────────────────────────────
print('[LOAD] pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
pipeline.cuda()
flow_model = pipeline.models['slat_flow_model']
dec_model  = pipeline.models['slat_decoder_mesh']
dino_model = pipeline.models['image_cond_model']
for m in [flow_model, dec_model, dino_model]:
    for p in m.parameters(): p.requires_grad_(False)

renderer = make_renderer()
EXT  = EXTRINSICS.to(DEVICE)
INTR = INTRINSICS.to(DEVICE)
_layouts = dec_model.mesh_extractor.layouts
COLOR_START, COLOR_END = _layouts['color']['range']
print(f'[LAYOUT] color: {COLOR_START}:{COLOR_END}')

# ── DINOv2 encode frame 75 (correct: is_training + x_prenorm + layer_norm) ───
_dino_tfm = T.Compose([
    T.Resize((518, 518), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print(f'[DINO] Encoding frame {FRAME_IDX}...')
img = Image.open(GT_FRAMES_DIR / f'frame_{FRAME_IDX:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
t   = _dino_tfm(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    feats = dino_model(t, is_training=True)['x_prenorm']
    toks  = F.layer_norm(feats, feats.shape[-1:])
cond75 = toks  # shape: (1, 257, 1024)

# ── sparse structure ───────────────────────────────────────────────────────────
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

# ── denoise frame 75 (25-step flow) ───────────────────────────────────────────
print('[SLAT] Denoising frame 75...')
ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
with torch.no_grad():
    for t_val, t_prev in T_PAIRS:
        t_ten = torch.tensor([1000.0 * t_val], device=DEVICE, dtype=torch.float32)
        v  = flow_model(ns, t_ten, cond75)
        ns = ns.replace(ns.feats - (t_val - t_prev) * v.feats)
slat75 = normalize_slat(ns)
ff75, fc75 = slat75.feats.cpu(), slat75.coords.cpu()
flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
print(f'[SLAT] done  feats={ff75.shape}')

# ── load LoRA ─────────────────────────────────────────────────────────────────
cfg  = json.loads((RUN_DIR / 'config.json').read_text())
RANK = cfg.get('rank', 4)
active = cfg.get('active_blocks', list(range(12)))
registry = None
if active:
    best = RUN_DIR / 'lora_ckpts' / 'lora_best.pt'
    candidates = sorted(RUN_DIR.glob('lora_ckpts/lora_e*.pt'))
    ckpt_path = best if best.exists() else (candidates[-1] if candidates else None)
    if ckpt_path:
        print(f'[CKPT] {ckpt_path.name}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'registry_state' in ckpt:
            registry = DecLoRARegistry(active, rank=RANK)
            registry.load_state_dict(ckpt['registry_state'])
            registry.to(DEVICE).eval()
            print(f'[CKPT] LoRA loaded  active={active}  rank={RANK}')

# ── canonical geometry (frozen decode of frame 75) ────────────────────────────
dec_model.to(DEVICE)
print('[CANONICAL] Frozen decode frame 75...')
canonical_feats = {}
def _cap(mod, inp, out): canonical_feats['v'] = out.feats.clone()
h = dec_model.out_layer.register_forward_hook(_cap)
st75 = sp.SparseTensor(feats=ff75.to(DEVICE), coords=fc75.to(DEVICE))
with torch.no_grad(): dec_model(st75)
h.remove()
canonical_out = canonical_feats['v']
print(f'[CANONICAL] shape={canonical_out.shape}')

# ── render helpers ─────────────────────────────────────────────────────────────
def filter_degen(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]; return mesh

def render(mesh):
    res  = renderer.render(mesh, EXT, INTR, return_types=['color', 'mask'])
    mask = res['mask'].unsqueeze(0)
    col  = res['color'] * mask + (1.0 - mask)
    return (col.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

print('[RENDER] Frozen...')
with torch.no_grad():
    frozen_arr = render(filter_degen(dec_model(st75)[0]))

print('[RENDER] LoRA (canonical geom swap)...')
if registry is not None:
    def _hybrid(mod, inp, out):
        hybrid = canonical_out.clone().to(out.feats.dtype)
        hybrid[:, COLOR_START:COLOR_END] = out.feats[:, COLOR_START:COLOR_END]
        return out.replace(hybrid)
    h = dec_model.out_layer.register_forward_hook(_hybrid)
    with torch.no_grad():
        with dec_lora_ctx(dec_model, registry):
            lora_arr = render(filter_degen(dec_model(st75)[0]))
    h.remove()
else:
    lora_arr = frozen_arr.copy()

gt_arr = np.array(
    Image.open(GT_FRAMES_DIR / f'frame_{FRAME_IDX:04d}.png')
         .convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
)

# ── save strip ────────────────────────────────────────────────────────────────
W, H = gt_arr.shape[1], gt_arr.shape[0]
LABEL_H = 28
canvas = Image.new('RGB', (W * 3, H + LABEL_H), (15, 15, 15))
draw   = ImageDraw.Draw(canvas)
for i, (arr, lbl, col) in enumerate([
    (gt_arr,     'GT video',             (80, 200, 80)),
    (frozen_arr, 'Frozen baseline',      (200, 80, 80)),
    (lora_arr,   f'LoRA dec-late r{RANK}', (80, 160, 255)),
]):
    canvas.paste(Image.fromarray(arr), (i * W, LABEL_H))
    draw.text((i * W + 8, 6), lbl, fill=col)

canvas.save(OUT_IMG)
print(f'[DONE] → {OUT_IMG}')
