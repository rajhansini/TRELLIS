"""
rung5_colonly_lora_v2.py
------------------------
Color-only decoder LoRA with L2 color-deviation regularization.

Problem with v1 (rung5_colonly_lora.py):
  Handle voxels barely visible in front-view renders → weak PSNR gradient.
  Transformer self-attention propagates body→lava signal to all voxels, pushing
  handle voxel raw_lora_col toward +10 to +15.
  hard clamp at ABS_MAX=8.0 → gradient = 0 at boundary → LoRA stuck outputting
  white handle → invisible on white background.

Fix:
  Compute reg on the RAW (unclamped) lora color output:
    reg = LAMBDA_REG * (raw_lora_col - frozen_col.detach()).pow(2).mean()
    total = render_loss + reg
  Gradient through reg is never zero (no clamp between raw_lora_col and the loss).
  Handle voxels (weak PSNR signal): reg dominates → stays near frozen (dark).
  Body voxels (strong PSNR signal at lava frames): PSNR dominates → learns lava colors.
  Gradient always flows → no stuck boundary.

LAMBDA_REG=0.01: body PSNR gradient >> 2*0.01*14=0.28 (body delta ~14 units).
               handle PSNR gradient << reg → handle stays near frozen.

Same two-pass colonly forward at train and inference → no OOD for FlexiCubes.
"""

import sys, os, argparse as _ap, hashlib, json
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent.parent
_PIPE  = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_RUNS  = _HERE / 'runs'
_RUNS.mkdir(parents=True, exist_ok=True)

_ap_ = _ap.ArgumentParser()
_ap_.add_argument('--rank',       type=int,   default=4)
_ap_.add_argument('--seed',       type=int,   default=6)
_ap_.add_argument('--epochs',     type=int,   default=30)
_ap_.add_argument('--abs-min',    type=float, default=-9.0)
_ap_.add_argument('--abs-max',    type=float, default=8.0)
_ap_.add_argument('--lambda-reg', type=float, default=0.01,
                  help='L2 penalty weight on (raw_lora_col - frozen_col)^2')
_ap_.add_argument('--smoke',      action='store_true')
_ap_.add_argument('--diag-every', type=int,   default=5)
args = _ap_.parse_args()

ACTIVE_BLOCKS = list(range(12))

# ── constants ──────────────────────────────────────────────────────────────────
GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED   = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES     = 150
STRUCT_SEED  = 42
STEPS        = 25
RESCALE_T    = 3.0
LR           = 1e-4
LOSS_SCALE0  = 4096.0
GRAD_CLIP    = 1.0
W_LPIPS      = 0.1
DEC_DIM      = 768
COLOR_START  = 53
COLOR_END    = 101
ABS_MIN      = args.abs_min
ABS_MAX      = args.abs_max
LAMBDA_REG   = args.lambda_reg

HELD_OUT = list(range(5, 151, 10))
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

# ── run identity ───────────────────────────────────────────────────────────────
_CFG = dict(
    variant='color_only_reg',
    rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    abs_min=ABS_MIN, abs_max=ABS_MAX, lambda_reg=LAMBDA_REG,
    held_out=HELD_OUT, active_blocks=ACTIVE_BLOCKS,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = f'rung5_colonyreg_r{args.rank}_s{args.seed}_{RUN_ID}'
_OUT   = _RUNS / _LABEL
_CKPT  = _OUT / 'lora_ckpts'
_DIAG  = _OUT / 'diag_renders'

for _d in (_OUT, _CKPT, _DIAG):
    _d.mkdir(parents=True, exist_ok=True)

class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(_OUT / 'train.log')
sys.stderr = sys.stdout

# ── nvdiffrast arch guard ──────────────────────────────────────────────────────
def _ensure_nvdiffrast():
    import subprocess, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm{p.major}{p.minor}'
    arch_str = f'{p.major}.{p.minor}'
    local    = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)
    try:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext(); del glctx
        print('[NVDIFF] OK', flush=True); return
    except Exception as e:
        print(f'[NVDIFF] FAILED: {e}', flush=True)
    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')
    print(f'[NVDIFF] building for {arch_tag} -> {local}', flush=True)
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    print('[NVDIFF] build done — restarting', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

# ─────────────────────────────────────────────────────────────────────────────
import math, time, gc, random
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as _ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM  = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]


# ── LoRA modules ───────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class DecBlockLoRABundle(nn.Module):
    def __init__(self, rank: int, dim: int = DEC_DIM):
        super().__init__()
        mlp_h = dim * 4
        self.lora_qkv = LoRALayer(dim, 3 * dim, rank)
        self.lora_out = LoRALayer(dim, dim,     rank)
        self.lora_fc1 = LoRALayer(dim, mlp_h,  rank)
        self.lora_fc2 = LoRALayer(mlp_h, dim,  rank)


class DecLoRARegistry(nn.Module):
    def __init__(self, active_blocks, rank: int):
        super().__init__()
        self.active = set(active_blocks)
        self.blocks = nn.ModuleDict({
            str(i): DecBlockLoRABundle(rank=rank) for i in active_blocks
        })

    def get(self, block_idx: int):
        key = str(block_idx)
        return self.blocks[key] if key in self.blocks else None


@contextmanager
def dec_block_lora_ctx(dec_model, registry: DecLoRARegistry):
    handles = []
    for i, block in enumerate(dec_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue
        def _qkv_hook(mod, inp, out, _lb=lb):
            return out + _lb.lora_qkv(inp[0]).to(out.dtype)
        def _out_hook(mod, inp, out, _lb=lb):
            return out + _lb.lora_out(inp[0]).to(out.dtype)
        def _fc1_hook(mod, inp, out, _lb=lb):
            d = _lb.lora_fc1(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))
        def _fc2_hook(mod, inp, out, _lb=lb):
            d = _lb.lora_fc2(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))
        handles.append(block.attn.to_qkv.register_forward_hook(_qkv_hook))
        handles.append(block.attn.to_out.register_forward_hook(_out_hook))
        handles.append(block.mlp.mlp[0].register_forward_hook(_fc1_hook))
        handles.append(block.mlp.mlp[2].register_forward_hook(_fc2_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ── Color-only two-pass forward ────────────────────────────────────────────────

def colonly_forward(dec_model, registry, slat_norm, with_grad=True, return_feats=False):
    """
    Two-pass decode: frozen geometry + LoRA colors (abs clamped).

    Pass 1 (no_grad):  frozen decoder → capture out_layer → geometry channels (0:53)
    Pass 2:            LoRA decoder   → capture out_layer → color channels (53:101)
    Mix: cat([frozen_geom.detach(), raw_lora_col.clamp(ABS_MIN, ABS_MAX)])
    Run: to_representation(mixed) → mesh

    return_feats=True (training only): also returns (raw_lora_col, frozen_col) for
    computing the L2 regularization on the unclamped LoRA output.
    """
    captured = {}

    def _hook(key):
        def _fn(mod, inp, out):
            captured[key] = out
        return _fn

    # Pass 1: frozen (always no_grad)
    with torch.no_grad():
        h1 = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        dec_model(slat_norm)
        h1.remove()

    # Pass 2: LoRA
    _ctx = torch.no_grad() if not with_grad else _nullctx()
    with _ctx:
        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_block_lora_ctx(dec_model, registry):
            dec_model(slat_norm)
        h2.remove()

    frozen_h = captured['frozen']
    lora_h   = captured['lora']

    frozen_geom = frozen_h.feats[:, :COLOR_START].detach()
    # Raw unclamped LoRA colors — used for reg loss (gradient always flows here)
    raw_lora_col = lora_h.feats[:, COLOR_START:COLOR_END]
    # Clamped colors go into the mesh extractor
    lora_col_clamped = raw_lora_col.clamp(ABS_MIN, ABS_MAX)

    mixed_feats = torch.cat([frozen_geom, lora_col_clamped], dim=1)
    mixed_h     = frozen_h.replace(mixed_feats)

    _ctx2 = torch.no_grad() if not with_grad else _nullctx()
    with _ctx2:
        meshes = dec_model.to_representation(mixed_h)

    mesh = meshes[0]
    if return_feats:
        frozen_col = frozen_h.feats[:, COLOR_START:COLOR_END].detach()
        return mesh, raw_lora_col, frozen_col
    return mesh


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx):
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt(frame_idx):
    img    = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img    = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt     = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = (gt < 0.99).any(dim=0)
    return gt, gt_mask


def load_render_mask():
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)


def masked_psnr(pred, gt, mask):
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred, gt):
    p = pred.permute(1, 2, 0).cpu().numpy()
    g = gt.permute(1, 2, 0).cpu().numpy()
    return float(_ssim(p, g, data_range=1.0, channel_axis=2))


def masked_loss(rendered, gt, render_mask, gt_mask, lpips_fn):
    m   = (render_mask | gt_mask).float()
    mse = ((rendered - gt) ** 2 * m).sum() / (m.sum() * 3 + 1e-8)
    r   = rendered * m + (1 - m)
    g   = gt       * m + (1 - m)
    lp  = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).mean()
    return mse, lp, mse + W_LPIPS * lp


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl):
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(ns, t_ten, cond_gl)
            ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


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
    return res['color'] * mask + (1.0 - mask), mask


def b_grad_norms(registry):
    norms = []
    for blk in registry.blocks.values():
        for lyr in (blk.lora_qkv, blk.lora_out, blk.lora_fc1, blk.lora_fc2):
            g = lyr.B.grad
            if g is not None:
                norms.append(g.norm().item())
    return norms


def b_norms(registry):
    norms = []
    for blk in registry.blocks.values():
        for lyr in (blk.lora_qkv, blk.lora_out, blk.lora_fc1, blk.lora_fc2):
            norms.append(lyr.B.float().norm().item())
    return norms


def _slat_from_cache(slat_cache, fi):
    feats, coords = slat_cache[fi]
    return sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))


def precompute_slats(flow_model, raw_tokens, noise_feats, coords, frame_list):
    print(f'\n[SLAT CACHE] Precomputing {len(frame_list)} SLaTs...', flush=True)
    cache = {}
    for k, fi in enumerate(frame_list):
        cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat = full_denoise_nograd(flow_model, noise_feats, coords, cond)
        cache[fi] = (slat.feats.cpu(), slat.coords.cpu())
        del slat, cond
        if (k + 1) % 25 == 0 or k == len(frame_list) - 1:
            print(f'  {k+1}/{len(frame_list)}', flush=True)
    gc.collect(); torch.cuda.empty_cache()
    print('[SLAT CACHE] done.', flush=True)
    return cache


# ── Diagnostics ────────────────────────────────────────────────────────────────

def make_strip(panels, cell=320, label_h=28):
    canvas = Image.new('RGB', (cell * len(panels), cell + label_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (img_src, lbl) in enumerate(panels):
        if isinstance(img_src, (str, Path)):
            img = Image.open(img_src).convert('RGB').resize((cell, cell), Image.LANCZOS)
        else:
            arr = (img_src.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((cell, cell), Image.LANCZOS)
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col * cell, 0, (col + 1) * cell - 1, label_h - 1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0, 0), lbl)[2]
        except: tw = len(lbl) * 7
        draw.text((col * cell + (cell - tw) // 2, 6), lbl, fill=(210, 210, 210))
    return canvas


def save_diagnostics(dec_model, registry, slat_cache, renderer, epoch, baseline_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    dec_model.eval()
    for fi in DIAG_FRAMES:
        slat_norm = _slat_from_cache(slat_cache, fi)
        mesh = colonly_forward(dec_model, registry, slat_norm, with_grad=False)
        color, _ = render_mesh(mesh, renderer)
        render = color.detach().clamp(0, 1)
        if fi not in baseline_cache:
            with torch.no_grad():
                meshes_base = dec_model(slat_norm)
            color_base, _ = render_mesh(meshes_base[0], renderer)
            baseline_cache[fi] = color_base.detach().clamp(0, 1)
        strip = make_strip([
            (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
            (baseline_cache[fi],                    'frozen decoder'),
            (render,                                f'colonyreg e{epoch:03d}'),
        ])
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'render_f{fi:04d}.png')
        del render, mesh, color
        gc.collect(); torch.cuda.empty_cache()


def save_curves(history):
    if len(history) < 2:
        return
    ep   = [r['epoch']       for r in history]
    tot  = [r['loss_total']  for r in history]
    mse_ = [r['loss_mse']    for r in history]
    lp_  = [r['loss_lpips']  for r in history]
    reg_ = [r['loss_reg']    for r in history]
    psnr = [r['held_psnr']   for r in history]
    bn   = [r['B_norm_mean'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Rung 5 colonyreg — {_LABEL}', fontsize=11)
    axes[0].plot(ep, mse_, 'o-', color='#e06c75', lw=2, ms=4, label='MSE')
    axes[0].plot(ep, lp_,  's-', color='#d19a66', lw=2, ms=4, label='LPIPS×0.1')
    axes[0].plot(ep, reg_, 'd-', color='#56b6c2', lw=2, ms=4, label=f'reg×{LAMBDA_REG}')
    axes[0].plot(ep, tot,  '^-', color='#c678dd', lw=2, ms=4, label='total')
    axes[0].set_title('Train loss'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, psnr, 'o-', color='#98c379', lw=2, ms=4)
    axes[1].set_title('Held-out masked PSNR (dB)'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, bn, 'o-', color='#61afef', lw=2, ms=4)
    axes[2].set_title('Mean ||B||'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_frames(dec_model, registry, slat_cache, renderer, render_mask, lpips_fn, frame_list):
    dec_model.eval()
    per_frame = []
    for fi in frame_list:
        slat_norm = _slat_from_cache(slat_cache, fi)
        mesh  = colonly_forward(dec_model, registry, slat_norm, with_grad=False)
        color, _ = render_mesh(mesh, renderer)
        render = color.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)
        psnr = masked_psnr(render, gt, render_mask)
        ssim = compute_ssim(render, gt)
        m = (render_mask | gt_mask).float()
        r = render * m + (1 - m); g = gt * m + (1 - m)
        with torch.no_grad():
            lp = lpips_fn(r.unsqueeze(0) * 2 - 1,
                          g.unsqueeze(0) * 2 - 1).item()
        per_frame.append({'frame': fi, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
        del render, gt, gt_mask, mesh, color
        gc.collect(); torch.cuda.empty_cache()
    psnrs = [r['psnr']  for r in per_frame]
    ssims = [r['ssim']  for r in per_frame]
    lpips = [r['lpips'] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'lpips_mean': float(np.mean(lpips)), 'lpips_std': float(np.std(lpips)),
        'per_frame': per_frame,
    }


def find_latest_ckpt():
    ckpts = sorted(_CKPT.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    print('=' * 72)
    print(f'Rung 5 Color-Only Decoder LoRA  v2 (with reg)')
    print(f'  active_blocks : {ACTIVE_BLOCKS}')
    print(f'  rank          : {args.rank}')
    print(f'  seed          : {args.seed}')
    print(f'  epochs        : {EPOCHS}')
    print(f'  abs_clamp     : [{ABS_MIN}, {ABS_MAX}]')
    print(f'  lambda_reg    : {LAMBDA_REG}')
    print(f'  run_id        : {RUN_ID}')
    print(f'  output        : {_OUT}')
    print('=' * 72)

    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL},
              open(_OUT / 'config.json', 'w'), indent=2)

    assert len(set(HELD_OUT) & set(TRAIN)) == 0

    # ── pipeline ──────────────────────────────────────────────────────────────
    print(f'\n[LOAD] {PRETRAINED}')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']

    N_DEC_actual = len(dec_model.blocks)
    assert N_DEC_actual == 12, f'N_DEC={N_DEC_actual} != 12'

    for p in flow_model.parameters(): p.requires_grad_(False)
    for p in dec_model.parameters():  p.requires_grad_(False)

    # ── structure ──────────────────────────────────────────────────────────────
    print(f'\n[STRUCT] seed={STRUCT_SEED}', flush=True)
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}', flush=True)
    assert N_vox == 7301, f'N_vox={N_vox} != 7301'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA registry ──────────────────────────────────────────────────────────
    registry = DecLoRARegistry(ACTIVE_BLOCKS, rank=args.rank).to(DEVICE)
    n_params = sum(p.numel() for p in registry.parameters())
    print(f'\n[LORA] blocks={ACTIVE_BLOCKS}  rank={args.rank}  params={n_params:,}')

    expected = args.rank * 12288 * len(ACTIVE_BLOCKS)
    assert n_params == expected, f'GATE 0 FAILED: {n_params} != {expected}'
    print(f'[GATE 0] PASSED ({n_params:,} params)\n')

    # ── fixed noise ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # ── DINOv2 cache ───────────────────────────────────────────────────────────
    print(f'[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    # ── SLaT cache (disk-backed for resume) ───────────────────────────────────
    all_frames       = list(range(1, N_FRAMES + 1))
    _slat_cache_disk = _OUT / 'slat_cache.npz'

    # Reuse v1 cache if present — same SLaTs (same struct seed, same noise seed)
    _v1_cache = _HERE / 'runs' / 'rung5_colonly_r4_s6_5d6b1700' / 'slat_cache.npz'

    if _slat_cache_disk.exists():
        print(f'\n[SLAT CACHE] Loading from disk (resume): {_slat_cache_disk}', flush=True)
        _npz      = np.load(_slat_cache_disk)
        _sarr     = _npz['slats']
        _coords_t = torch.from_numpy(_npz['coords']).int()
        slat_cache = {fi: (torch.from_numpy(_sarr[fi - 1]).float(), _coords_t)
                      for fi in range(1, N_FRAMES + 1)}
        flow_model.cpu()
        print(f'[SLAT CACHE] Loaded  frames={_sarr.shape[0]}  N_vox={_sarr.shape[1]}',
              flush=True)
    elif _v1_cache.exists():
        print(f'\n[SLAT CACHE] Reusing v1 cache: {_v1_cache}', flush=True)
        _npz      = np.load(_v1_cache)
        _sarr     = _npz['slats']
        _coords_t = torch.from_numpy(_npz['coords']).int()
        slat_cache = {fi: (torch.from_numpy(_sarr[fi - 1]).float(), _coords_t)
                      for fi in range(1, N_FRAMES + 1)}
        flow_model.cpu()
        # Save to local disk so resume works cleanly
        np.savez_compressed(_slat_cache_disk, slats=_sarr, coords=_npz['coords'])
        print(f'[SLAT CACHE] Copied to {_slat_cache_disk}', flush=True)
    else:
        slat_cache = precompute_slats(flow_model, raw_tokens,
                                      fixed_noise_feats, coords, all_frames)
        flow_model.cpu()
        _sarr = np.stack([slat_cache[fi][0].numpy() for fi in range(1, N_FRAMES + 1)])
        np.savez_compressed(_slat_cache_disk, slats=_sarr, coords=coords.cpu().numpy())
        print(f'\n[SLAT CACHE] Saved to disk → {_slat_cache_disk}', flush=True)

    gc.collect(); torch.cuda.empty_cache()
    dec_model.to(DEVICE)

    # ── LPIPS + mask + renderer ────────────────────────────────────────────────
    print('\n[LPIPS] Loading AlexNet...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)

    render_mask = load_render_mask()
    renderer    = make_renderer()

    # ── resume ─────────────────────────────────────────────────────────────────
    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None

    if not resumed:
        print('\n[GATE-plain] B=0 identity check (two-pass)...')
        slat_ref  = _slat_from_cache(slat_cache, 75)
        ext       = EXTRINSICS.to(DEVICE)
        intr      = INTRINSICS.to(DEVICE)
        with torch.no_grad():
            meshes_frz = dec_model(slat_ref)
        mesh_frz = filter_degenerate_faces(meshes_frz[0])
        res_frz  = renderer.render(mesh_frz, ext, intr, return_types=['color', 'mask'])
        mask_frz = res_frz['mask'].unsqueeze(0)
        color_frz = (res_frz['color'] * mask_frz + (1 - mask_frz)).detach()

        mesh_co  = colonly_forward(dec_model, registry, slat_ref, with_grad=False)
        mesh_co  = filter_degenerate_faces(mesh_co)
        res_co   = renderer.render(mesh_co, ext, intr, return_types=['color', 'mask'])
        mask_co  = res_co['mask'].unsqueeze(0)
        color_co = (res_co['color'] * mask_co + (1 - mask_co)).detach()

        diff = (color_frz - color_co).abs().max().item()
        print(f'  max |frozen - colonly(B=0)| = {diff:.3e}  (tol=1e-2)')
        assert diff < 1e-2, f'GATE-plain FAILED: {diff:.3e}'
        print('[GATE-plain] PASSED\n')
        del mesh_frz, mesh_co, color_frz, color_co
        gc.collect(); torch.cuda.empty_cache()
    else:
        print('\n[GATES] SKIPPED — resuming from checkpoint\n')

    # ── optimizer + resume ─────────────────────────────────────────────────────
    trainable_params = list(registry.parameters())
    optimizer   = torch.optim.Adam(trainable_params, lr=LR, eps=1e-16, weight_decay=0.0)
    start_epoch = 1
    history     = []
    best_psnr   = -float('inf')

    if resumed:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        assert ckpt.get('run_id') == RUN_ID, \
            f'run_id mismatch: {ckpt.get("run_id")} != {RUN_ID}'
        registry.load_state_dict(ckpt['registry_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -float('inf'))
        loss_scale  = ckpt.get('loss_scale', LOSS_SCALE0)
        hist_path   = _OUT / 'loss_history.json'
        if hist_path.exists():
            history = json.load(open(hist_path))
        print(f'  resumed from epoch {ckpt["epoch"]}  loss_scale={loss_scale:.0f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    if not resumed:
        print('\n[GATE 2] Gradient flow check...', flush=True)
        dec_model.eval()
        slat_75 = _slat_from_cache(slat_cache, 75)
        optimizer.zero_grad()
        mesh, raw_lc, frz_c = colonly_forward(
            dec_model, registry, slat_75, with_grad=True, return_feats=True)
        color, _ = render_mesh(mesh, renderer)
        gt_75, gm_75 = load_gt(75)
        _, _, loss_g2 = masked_loss(color, gt_75, render_mask, gm_75, lpips_fn)
        reg_g2 = LAMBDA_REG * (raw_lc - frz_c).pow(2).mean()
        ((loss_g2 + reg_g2) * loss_scale).backward()

        gnorms  = b_grad_norms(registry)
        n_total = len(ACTIVE_BLOCKS) * 4
        n_zero  = sum(1 for g in gnorms if g == 0.0)
        print(f'  B.grad norms ({len(gnorms)}/{n_total}): '
              f'min={min(gnorms):.3e}  max={max(gnorms):.3e}', flush=True)
        if n_zero:
            print(f'  WARNING: {n_zero}/{n_total} B matrices have zero grad '
                  f'(fp16 early blocks; expected)', flush=True)
        assert len(gnorms) == n_total, \
            f'GATE 2 FAILED: {len(gnorms)}/{n_total} B mats got grad'
        optimizer.zero_grad()
        print('[GATE 2] PASSED\n', flush=True)
        del mesh, color, gt_75, gm_75, loss_g2, reg_g2, raw_lc, frz_c
        gc.collect(); torch.cuda.empty_cache()
    else:
        print('[GATE 2] SKIPPED — resuming\n', flush=True)

    # ── Training ───────────────────────────────────────────────────────────────
    dec_model.eval()
    baseline_cache = {}

    print(f'[TRAIN] epochs {start_epoch}→{EPOCHS}  frames/epoch={len(TRAIN)}'
          f'  LOSS_SCALE={loss_scale:.0f}  lambda_reg={LAMBDA_REG}', flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        ep_t0 = time.time()
        frame_order = TRAIN[:]
        random.shuffle(frame_order)
        tot_loss = tot_mse = tot_lp = tot_reg = 0.0
        n_steps  = 0

        print(f'\n[EPOCH {epoch}/{EPOCHS}] starting...', flush=True)

        for fi in frame_order:
            slat_norm = _slat_from_cache(slat_cache, fi)
            optimizer.zero_grad()

            # Two-pass: frozen geometry + LoRA colors
            # return_feats=True gives raw (unclamped) lora_col and frozen_col for reg
            mesh, raw_lora_col, frozen_col = colonly_forward(
                dec_model, registry, slat_norm, with_grad=True, return_feats=True)
            color, _ = render_mesh(mesh, renderer)
            gt, gt_mask = load_gt(fi)
            mse, lp, render_loss = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)

            # L2 regularization on unclamped LoRA colors vs frozen
            # Gradient flows through raw_lora_col regardless of clamp saturation.
            reg = LAMBDA_REG * (raw_lora_col - frozen_col).pow(2).mean()

            total = render_loss + reg
            (total * loss_scale).backward()
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.div_(loss_scale)
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()

            tot_loss += total.item()
            tot_mse  += mse.item()
            tot_lp   += lp.item()
            tot_reg  += reg.item()
            n_steps  += 1

            if n_steps % 15 == 0:
                print(f'  e{epoch:02d} [{n_steps:03d}/{len(TRAIN)}] f{fi:03d}  '
                      f'mse={mse.item():.5f}  lpips={lp.item():.5f}  '
                      f'reg={reg.item():.5f}  total={total.item():.5f}', flush=True)

            del mesh, color, gt, gt_mask, mse, lp, render_loss, reg, total
            del raw_lora_col, frozen_col, slat_norm
            gc.collect(); torch.cuda.empty_cache()

        # Epoch eval
        held_res  = evaluate_frames(dec_model, registry, slat_cache, renderer,
                                    render_mask, lpips_fn, HELD_OUT)
        held_psnr = held_res['psnr_mean']
        held_std  = held_res['psnr_std']
        held_ssim = held_res['ssim_mean']
        held_lp   = held_res['lpips_mean']

        bnorms  = b_norms(registry)
        ep_loss = tot_loss / max(n_steps, 1)
        ep_mse  = tot_mse  / max(n_steps, 1)
        ep_lp   = tot_lp   / max(n_steps, 1)
        ep_reg  = tot_reg  / max(n_steps, 1)
        ep_time = time.time() - ep_t0
        new_best = held_psnr > best_psnr
        if new_best:
            best_psnr = held_psnr

        print(f'\n[EPOCH {epoch}/{EPOCHS}]  loss={ep_loss:.5f}  '
              f'(mse={ep_mse:.5f}  lpips×0.1={ep_lp * W_LPIPS:.5f}  '
              f'reg={ep_reg:.5f})', flush=True)
        print(f'  ||B||  mean={np.mean(bnorms):.4f}  '
              f'min={np.min(bnorms):.4f}  max={np.max(bnorms):.4f}', flush=True)
        print(f'  held  PSNR={held_psnr:.3f}±{held_std:.3f}  '
              f'SSIM={held_ssim:.4f}  LPIPS={held_lp:.4f}  '
              f'GPU={torch.cuda.max_memory_allocated()/1e9:.1f}GB  '
              + ('BEST ✓' if new_best else '') + f'  time={ep_time:.1f}s', flush=True)

        ckpt = {
            'epoch': epoch, 'run_id': RUN_ID, 'best_psnr': best_psnr,
            'loss_scale': loss_scale, 'optimizer': optimizer.state_dict(),
            'registry_state': registry.state_dict(),
        }
        torch.save(ckpt, _CKPT / f'lora_e{epoch:03d}.pt')
        if new_best:
            torch.save(ckpt, _CKPT / 'lora_best.pt')
            print(f'  [CKPT] new best → lora_best.pt')

        rec = {
            'epoch': epoch, 'loss_total': ep_loss, 'loss_mse': ep_mse,
            'loss_lpips': ep_lp, 'loss_reg': ep_reg, 'held_psnr': held_psnr,
            'held_std': held_std, 'held_ssim': held_ssim, 'held_lpips': held_lp,
            'B_norm_mean': float(np.mean(bnorms)), 'B_norm_max': float(np.max(bnorms)),
            'time_s': ep_time,
        }
        history.append(rec)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)
        save_curves(history)

        if epoch % DIAG_EVERY == 0:
            save_diagnostics(dec_model, registry, slat_cache, renderer,
                             epoch, baseline_cache)

    # ── Final eval ─────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print('[FINAL EVAL] All 150 frames')
    print('='*72)

    best_ckpt_path = _CKPT / 'lora_best.pt'
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True)
        registry.load_state_dict(best_ckpt['registry_state'])
        print(f'  Loaded best (epoch={best_ckpt["epoch"]}  '
              f'psnr={best_ckpt["best_psnr"]:.3f})')

    res_held  = evaluate_frames(dec_model, registry, slat_cache, renderer,
                                render_mask, lpips_fn, HELD_OUT)
    res_train = evaluate_frames(dec_model, registry, slat_cache, renderer,
                                render_mask, lpips_fn, TRAIN)
    res_all   = evaluate_frames(dec_model, registry, slat_cache, renderer,
                                render_mask, lpips_fn, all_frames)

    print(f'  All:   PSNR={res_all["psnr_mean"]:.3f}±{res_all["psnr_std"]:.3f}  '
          f'SSIM={res_all["ssim_mean"]:.4f}  LPIPS={res_all["lpips_mean"]:.4f}')
    print(f'  Held:  PSNR={res_held["psnr_mean"]:.3f}±{res_held["psnr_std"]:.3f}')
    print(f'  Train: PSNR={res_train["psnr_mean"]:.3f}±{res_train["psnr_std"]:.3f}')

    json.dump({'run_id': RUN_ID, 'held': res_held, 'train': res_train, 'all': res_all},
              open(_OUT / 'final_eval.json', 'w'), indent=2)
    print(f'\n[DONE]  best held PSNR={best_psnr:.3f} dB  output={_OUT}')


if __name__ == '__main__':
    main()
