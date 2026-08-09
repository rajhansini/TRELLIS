"""
Rung 2 — Placement ablation at matched capacity.

Question: does WHERE you place LoRA matter more than HOW MUCH capacity you give it?

Config-driven: --placement (early/mid/late/all) x --rank (4 = 2a screen, 12 = 2b matched)

Param math per block (dim=1024):
  to_q:  A(r,1024)+B(1024,r) = 2048r
  to_kv: A(r,1024)+B(2048,r) = 3072r
  total: 5120r per block

  2a: 8 blocks × rank 4  =  163,840 params
  2b: 8 blocks × rank 12 =  491,520 params  ==  Rung 1 (24 blocks × rank 4)

Output:
  lora_experiments/runs/rung2_{placement}_r{rank}_s{seed}_{hash}/
    config.json           — full config + run_id
    loss_history.json     — per-epoch metrics
    lora_ckpts/           — lora_e{N:03d}.pt, lora_best.pt
    diag_renders/         — e{N:03d}/strip_f{F:04d}.png
    train.log
  lora_experiments/rung2_gate3/
    init_{placement}.pt   — fixed-seed init render, used for Gate 3 byte-comparison
"""

import sys, os, argparse as _ap, hashlib, json
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE    = Path(__file__).resolve().parent          # lora_experiments/
_ROOT    = _HERE.parent.parent                      # TRELLIS/
_PIPE    = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_RUNS    = _HERE / 'runs'
_GATE3   = _HERE / 'rung2_gate3'
_RUNS.mkdir(parents=True, exist_ok=True)
_GATE3.mkdir(exist_ok=True)


# ── parse args BEFORE any heavy imports ──────────────────────────────────────

_ap_ = _ap.ArgumentParser()
_ap_.add_argument('--placement', choices=['early', 'mid', 'late', 'all'], required=True)
_ap_.add_argument('--rank',      type=int, default=4)
_ap_.add_argument('--seed',      type=int, default=6)
_ap_.add_argument('--epochs',    type=int, default=15)
_ap_.add_argument('--smoke',     action='store_true')
_ap_.add_argument('--diag_every',type=int, default=1)
args = _ap_.parse_args()

BLOCK_RANGES = {
    'early': list(range(0,  8)),   # blocks 0-7
    'mid':   list(range(8,  16)),  # blocks 8-15
    'late':  list(range(16, 24)),  # blocks 16-23
    'all':   list(range(0,  24)),  # reproduces Rung 1
}
ACTIVE_BLOCKS = BLOCK_RANGES[args.placement]
N_ACTIVE      = len(ACTIVE_BLOCKS)

# ── constants ─────────────────────────────────────────────────────────────────

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH   = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES    = 150
STRUCT_SEED = 42
STEPS       = 25
RESCALE_T   = 3.0
LR          = 1e-4
LOSS_SCALE0 = 4096.0
GRAD_CLIP   = 1.0
W_LPIPS     = 0.1

HELD_OUT = list(range(5, 151, 10))    # [5,15,...,145]  15 frames, never trained
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

# ── run identity ──────────────────────────────────────────────────────────────

_CFG = dict(
    placement=args.placement, rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    held_out=HELD_OUT, blocks=ACTIVE_BLOCKS,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = f'rung2_{args.placement}_r{args.rank}_s{args.seed}_{RUN_ID}'
_OUT   = _RUNS / _LABEL
_CKPT  = _OUT / 'lora_ckpts'
_DIAG  = _OUT / 'diag_renders'

_OUT.mkdir(parents=True, exist_ok=True)
_CKPT.mkdir(exist_ok=True)
_DIAG.mkdir(exist_ok=True)

# ── tee stdout to log ─────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(_OUT / 'train.log')
sys.stderr = sys.stdout


# ── nvdiffrast arch guard ─────────────────────────────────────────────────────

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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from contextlib import contextmanager
from skimage.metrics import structural_similarity as _ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, decode_and_render, RENDER_RES,
)

DEVICE = torch.device('cuda')

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """fp32 forward, kaiming A, zero B, scaling=lora_alpha/rank."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.scaling = 1.0      # lora_alpha = rank  →  scaling = 1.0
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (((x.float() @ self.A.T) @ self.B.T) * self.scaling).to(x.dtype)


class LoRABlock(nn.Module):
    def __init__(self, dim: int = 1024, rank: int = 4):
        super().__init__()
        self.lora_q  = LoRALayer(dim, dim,     rank)
        self.lora_kv = LoRALayer(dim, 2 * dim, rank)


class LoRARegistry(nn.Module):
    """Adapters exist ONLY for active blocks; inactive blocks run frozen, untouched."""
    def __init__(self, active_blocks, dim=1024, rank=4):
        super().__init__()
        self.active = set(active_blocks)
        self.blocks = nn.ModuleDict({
            str(i): LoRABlock(dim=dim, rank=rank) for i in active_blocks
        })

    def get(self, block_idx):
        key = str(block_idx)
        return self.blocks[key] if key in self.blocks else None


def freeze_trellis(flow_model) -> None:
    for p in flow_model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in flow_model.parameters())
    print(f'[FREEZE] Froze {n:,} TRELLIS params')


# ── Attention ─────────────────────────────────────────────────────────────────

_ATTN_CHUNK = 256

from torch.utils.checkpoint import checkpoint as _grad_ckpt

def _chunk_attn_fn(qc, k, v, scale):
    """Single chunk attention — wrapped by gradient checkpoint for active blocks."""
    A = torch.einsum('nhd,mhd->nhm', qc.float(), k.float()) * scale
    A = torch.softmax(A, dim=-1)
    return torch.einsum('nhm,mhd->nhd', A, v.float())

def _attn_chunked(q, k, v, scale, chunk=_ATTN_CHUNK):
    # Gradient checkpoint each chunk when inputs carry grad (active blocks).
    # Without checkpointing: 8 active blocks × 29 chunks × (qc+A_post) ≈ 6.7 GB of
    # saved tensors during the last-step forward-with-grad pass → OOM on 10.57 GB GPUs.
    # With checkpointing: only q, k, v (≈570 MB total) are retained; A matrices are
    # recomputed during backward at the cost of one extra attention pass.
    use_ckpt = any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor))
    outs = []
    for s in range(0, q.shape[0], chunk):
        e  = min(s + chunk, q.shape[0])
        qc = q[s:e]
        if use_ckpt:
            out = _grad_ckpt(_chunk_attn_fn, qc, k, v, scale, use_reentrant=False)
        else:
            out = _chunk_attn_fn(qc, k, v, scale)
        outs.append(out)
    return torch.cat(outs, dim=0)


def _placement_fwd(module, x, context, lb):
    """
    lb is None  → inactive block, run frozen to_q/to_kv unchanged.
    lb is a LoRABlock → active block, add LoRA delta to q and kv.
    """
    ch  = module.channels
    H   = module.num_heads
    hd  = ch // H
    scale = hd ** -0.5
    wdt = module.to_q.weight.dtype

    x_feats = x.feats.to(wdt)
    ctx_2d  = (context.squeeze(0) if context.dim() == 3 else context).to(wdt)

    if lb is None:
        with torch.no_grad():
            q  = module.to_q(x_feats)
            kv = module.to_kv(ctx_2d)
    else:
        with torch.no_grad():
            q_base  = module.to_q(x_feats)
            kv_base = module.to_kv(ctx_2d)
        q  = q_base  + lb.lora_q(x_feats)
        kv = kv_base + lb.lora_kv(ctx_2d)

    k, v = kv.reshape(-1, 2, H, hd)[:, 0], kv.reshape(-1, 2, H, hd)[:, 1]
    q    = q.reshape(-1, H, hd)
    out  = _attn_chunked(q, k, v, scale)
    out  = out.to(wdt).reshape(-1, ch)
    out  = module.to_out(out)
    return x.replace(out.to(x.feats.dtype))


@contextmanager
def placement_ctx(flow_model, lora: LoRARegistry):
    """Patch all cross-attn blocks; active ones get LoRA, inactive run frozen."""
    saved   = {}
    ca_idx  = 0
    for blk in flow_model.blocks:
        if not hasattr(blk, 'cross_attn'):
            continue
        ca           = blk.cross_attn
        saved[ca_idx] = ca.forward
        lb            = lora.get(ca_idx)
        idx_          = ca_idx

        def _make(mod, lb_, idx_):
            def _fwd(x, context=None):
                return _placement_fwd(mod, x, context, lb_)
            return _fwd

        ca.forward = _make(ca, lb, idx_)
        ca_idx += 1

    assert ca_idx == 24, f'Expected 24 cross-attn blocks, found {ca_idx}'
    try:
        yield
    finally:
        ca_idx = 0
        for blk in flow_model.blocks:
            if hasattr(blk, 'cross_attn') and ca_idx in saved:
                blk.cross_attn.forward = saved[ca_idx]
                ca_idx += 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt(frame_idx: int):
    img     = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img     = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt      = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = (gt < 0.99).any(dim=0)
    return gt, gt_mask


def load_render_mask() -> torch.Tensor:
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)


def masked_psnr(pred, gt, mask) -> float:
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
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


# ── Denoising ─────────────────────────────────────────────────────────────────

def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, lora):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with placement_ctx(flow_model, lora):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, lora, require_grad: bool):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    if require_grad:
        with placement_ctx(flow_model, lora):
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with placement_ctx(flow_model, lora):
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl, lora):
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with placement_ctx(flow_model, lora):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


# ── Norms ─────────────────────────────────────────────────────────────────────

def registry_norms(lora: LoRARegistry, w_q_per_block: dict):
    """
    Returns per-active-block dict: block_idx -> {B_norm, BA_norm, BA_ratio}.
    w_q_per_block: block_idx -> float (||W_q|| for that block).
    """
    out = {}
    for idx_str, blk in lora.blocks.items():
        idx = int(idx_str)
        wq  = w_q_per_block.get(idx, 1.0)
        for name, layer in [('q', blk.lora_q), ('kv', blk.lora_kv)]:
            A  = layer.A.float()
            B  = layer.B.float()
            ba = (B @ A).norm().item()
            out[f'{idx}_{name}'] = {
                'B_norm':   B.norm().item(),
                'BA_norm':  ba,
                'BA_ratio': ba / wq,
            }
    return out


def b_grad_norms(lora: LoRARegistry):
    norms = []
    for blk in lora.blocks.values():
        for layer in (blk.lora_q, blk.lora_kv):
            g = layer.B.grad
            if g is not None:
                norms.append(g.norm().item())
    return norms


# ── Diagnostics ───────────────────────────────────────────────────────────────

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


def save_diagnostics(pipeline, flow_model, lora, raw_tokens, coords,
                     fixed_noise_feats, renderer, epoch, raw_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    flow_model.eval()
    for fi in DIAG_FRAMES:
        cond_gl  = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat     = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl, lora)
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render   = color.detach().clamp(0, 1)
        if fi not in raw_cache:
            raw_cache[fi] = render.clone()
        strip = make_strip([
            (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
            (raw_cache[fi],                          'TRELLIS baseline'),
            (render,                                 f'{args.placement} e{epoch:03d}'),
        ])
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'render_f{fi:04d}.png')
        del render, slat, color
        gc.collect(); torch.cuda.empty_cache()
    flow_model.train()


def save_curves(history, label):
    if len(history) < 2:
        return
    ep   = [r['epoch']      for r in history]
    tot  = [r['loss_total']  for r in history]
    mse_ = [r['loss_mse']   for r in history]
    lp_  = [r['loss_lpips'] for r in history]
    psnr = [r['held_psnr']  for r in history]
    bn   = [r['B_norm_mean'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Rung 2 — {label}', fontsize=11)
    axes[0].plot(ep, mse_, 'o-', color='#e06c75', lw=2, ms=4, label='MSE')
    axes[0].plot(ep, lp_,  's-', color='#d19a66', lw=2, ms=4, label='LPIPS×0.1')
    axes[0].plot(ep, tot,  '^-', color='#c678dd', lw=2, ms=4, label='total')
    axes[0].set_title('Train loss'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, psnr, 'o-', color='#98c379', lw=2, ms=4)
    axes[1].set_title('Held-out masked PSNR (dB)'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, bn, 'o-', color='#61afef', lw=2, ms=4)
    axes[2].set_title('Mean ||B|| across active blocks'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_frames(pipeline, flow_model, lora, raw_tokens, coords,
                    fixed_noise_feats, renderer, render_mask, lpips_fn, frame_list):
    flow_model.eval()
    per_frame = []
    for fi in frame_list:
        cond_gl  = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat     = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl, lora)
        # Offload flow model before decode+render (same pattern as training loop)
        # to avoid OOM on GPUs <12 GB where flow+decoder can't coexist.
        flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        flow_model.to(DEVICE)
        render   = color.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)
        psnr = masked_psnr(render, gt, render_mask)
        ssim = compute_ssim(render, gt)
        m = (render_mask | gt_mask).float()
        r = render * m + (1 - m); g = gt * m + (1 - m)
        with torch.no_grad():
            lp = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).item()
        per_frame.append({'frame': fi, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
        del render, gt, gt_mask, slat, color
        gc.collect(); torch.cuda.empty_cache()
    flow_model.train()
    psnrs = [r['psnr']  for r in per_frame]
    ssims = [r['ssim']  for r in per_frame]
    lpips = [r['lpips'] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'lpips_mean': float(np.mean(lpips)), 'lpips_std': float(np.std(lpips)),
        'per_frame': per_frame,
    }


def evaluate_held_out(pipeline, flow_model, lora, raw_tokens, coords,
                      fixed_noise_feats, renderer, render_mask, lpips_fn):
    res = evaluate_frames(pipeline, flow_model, lora, raw_tokens, coords,
                          fixed_noise_feats, renderer, render_mask, lpips_fn, HELD_OUT)
    return (res['psnr_mean'], res['psnr_std'], res['lpips_mean'],
            res['ssim_mean'], res['ssim_std'], res)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def find_latest_ckpt():
    ckpts = sorted(_CKPT.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    # ── param math gate 0 ──────────────────────────────────────────────────────
    EXPECTED_PARAMS = N_ACTIVE * 5120 * args.rank
    # 5120r = (2048 + 3072) * r = to_q (2048r) + to_kv (3072r)

    print('=' * 72)
    print(f'Rung 2 — Placement ablation')
    print(f'  placement    : {args.placement}  blocks={ACTIVE_BLOCKS}')
    print(f'  rank         : {args.rank}')
    print(f'  seed         : {args.seed}')
    print(f'  epochs       : {EPOCHS}')
    print(f'  expected params: {EXPECTED_PARAMS:,}  ({N_ACTIVE} blocks × 5120 × {args.rank})')
    print(f'  run_id       : {RUN_ID}')
    print(f'  output       : {_OUT}')
    print('=' * 72)

    # save config
    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL},
              open(_OUT / 'config.json', 'w'), indent=2)

    # ── split sanity ───────────────────────────────────────────────────────────
    assert len(set(HELD_OUT) & set(TRAIN)) == 0
    assert len(HELD_OUT) == 15 and len(TRAIN) == 135

    # ── pipeline ──────────────────────────────────────────────────────────────
    print(f'\n[LOAD] {PRETRAINED}')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── structure (fixed STRUCT_SEED) ──────────────────────────────────────────
    print(f'\n[STRUCT] seed={STRUCT_SEED}')
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}')
    assert N_vox == 7301, f'N_vox={N_vox} != 7301 — struct seed wrong'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA registry ─────────────────────────────────────────────────────────
    print(f'\n[LORA] Building LoRARegistry: placement={args.placement} rank={args.rank}')
    freeze_trellis(flow_model)
    lora = LoRARegistry(ACTIVE_BLOCKS, dim=1024, rank=args.rank).to(DEVICE)

    # GATE 0: param count
    actual_params = sum(p.numel() for p in lora.parameters())
    print(f'\n[GATE 0]  expected={EXPECTED_PARAMS:,}  actual={actual_params:,}')
    assert actual_params == EXPECTED_PARAMS, \
        f'GATE 0 FAILED: {actual_params} != {EXPECTED_PARAMS}'
    assert len(lora.blocks) == N_ACTIVE, \
        f'GATE 0: len(lora.blocks)={len(lora.blocks)} != {N_ACTIVE}'
    for blk in lora.blocks.values():
        assert blk.lora_q.B.abs().max().item()  == 0.0, 'B not zero at init'
        assert blk.lora_kv.B.abs().max().item() == 0.0, 'B not zero at init'
    assert not any(p.requires_grad for p in flow_model.parameters()), \
        'frozen TRELLIS has requires_grad=True!'
    print('[GATE 0] PASSED\n')

    # ── collect per-block W_q norms for BA_ratio logging ──────────────────────
    w_q_per_block = {}
    ca_idx = 0
    for blk in flow_model.blocks:
        if hasattr(blk, 'cross_attn'):
            if ca_idx in lora.active:
                w_q_per_block[ca_idx] = blk.cross_attn.to_q.weight.float().norm().item()
            ca_idx += 1
    print(f'  W_q norms for active blocks: '
          + ', '.join(f'[{i}]={v:.4f}' for i, v in sorted(w_q_per_block.items())))

    # ── optimizer + resume ─────────────────────────────────────────────────────
    optimizer   = torch.optim.Adam(lora.parameters(), lr=LR, eps=1e-16, weight_decay=0.0)
    start_epoch = 1
    history     = []
    best_psnr   = -float('inf')

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None
    if latest_ckpt:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        assert ckpt.get('run_id') == RUN_ID, \
            f'checkpoint run_id mismatch: {ckpt.get("run_id")} != {RUN_ID}'
        lora.load_state_dict(ckpt['lora_state'])
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

    # ── fixed noise (SAME seed for all runs — control) ─────────────────────────
    print(f'\n[NOISE] seed={args.seed}')
    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # ── DINOv2 cache ──────────────────────────────────────────────────────────
    print(f'\n[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    # ── LPIPS + mask ──────────────────────────────────────────────────────────
    print('\n[LPIPS] Loading AlexNet...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    render_mask = load_render_mask()
    print(f'[MASK] {render_mask.sum().item()} masked pixels '
          f'({100*render_mask.float().mean():.1f}%)')

    renderer = make_renderer()
    pipeline.models['slat_decoder_mesh'].to(DEVICE)

    # ── GATE 1 + Gate 3 (fresh runs only) ─────────────────────────────────────
    if resumed:
        print('\n[GATE 1] SKIPPED — resuming (B ≠ 0 by design)\n')
    else:
        print('\n[GATE 1] Identity check (B=0 → placement forward == raw TRELLIS)...')
        _g1_cond = raw_tokens[75].unsqueeze(0).to(DEVICE)

        # raw TRELLIS
        _g1_raw  = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                v = flow_model(_g1_raw, t_ten, _g1_cond)
                _g1_raw = _g1_raw.replace(_g1_raw.feats - (t - t_prev) * v.feats)

        # placement LoRA (B=0)
        _g1_lora = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                with placement_ctx(flow_model, lora):
                    v = flow_model(_g1_lora, t_ten, _g1_cond)
                _g1_lora = _g1_lora.replace(_g1_lora.feats - (t - t_prev) * v.feats)

        diff = (_g1_raw.feats - _g1_lora.feats).abs().max().item()
        print(f'  max |raw - placement(B=0)| = {diff:.3e}')
        assert diff < 1e-2, f'GATE 1 FAILED: max diff = {diff:.3e}'
        print('[GATE 1] PASSED')

        # Gate 3: save init feats tensor for cross-run byte-comparison
        _g3_path = _GATE3 / f'init_{args.placement}.pt'
        torch.save(_g1_lora.feats.cpu(), _g3_path)
        print(f'[GATE 3] init feats saved → {_g3_path}')

        # GATE 3-cross: if other placement init files exist, verify near-identity.
        # Placements share fixed_noise_feats + B=0, so outputs match up to xformers
        # self-attn non-determinism (~1e-3). Diffs >> 0.05 signal a different seed.
        _G3_CROSS_TOL = 0.05
        _cur_feats  = torch.load(_g3_path, map_location='cpu', weights_only=True)
        _g3_checked = 0
        for _cmp_pl in ['early', 'mid', 'late']:
            if _cmp_pl == args.placement:
                continue
            _cmp_f = _GATE3 / f'init_{_cmp_pl}.pt'
            if _cmp_f.exists():
                _cmp_feats = torch.load(_cmp_f, map_location='cpu', weights_only=True)
                _diff = (_cur_feats - _cmp_feats).abs().max().item()
                assert _diff < _G3_CROSS_TOL, (
                    f'GATE 3-cross FAIL: init_{args.placement} vs init_{_cmp_pl} '
                    f'max diff = {_diff:.3e}  (tol={_G3_CROSS_TOL:.2f}; '
                    f'check --seed and N_vox match across placements)')
                print(f'[GATE 3-cross] PASSED: init_{args.placement} vs init_{_cmp_pl}'
                      f'  max_diff={_diff:.3e}  (< {_G3_CROSS_TOL:.2f})')
                _g3_checked += 1
        if _g3_checked == 0:
            print('[GATE 3-cross] PENDING: no other placement init files yet\n')
        else:
            print(f'[GATE 3-cross] {_g3_checked}/2 cross-checks passed\n')
        del _g1_raw, _g1_lora, _g1_cond, _cur_feats
        gc.collect(); torch.cuda.empty_cache()

    # ── GATE 2: gradient check (fresh runs only) ──────────────────────────────
    if resumed:
        print('[GATE 2] SKIPPED — resuming from checkpoint\n')
    else:
        print('[GATE 2] Gradient flow on active blocks (frame 75)...')
        flow_model.train()
        _g2_cond   = raw_tokens[75].unsqueeze(0).to(DEVICE)
        _g2_noise  = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        _g2_prefix = denoise_prefix_nograd(flow_model, _g2_noise, _g2_cond, lora)
        _g2_x0v    = denoise_last_step(flow_model, _g2_prefix, _g2_cond, lora, require_grad=False)
        _g2_sv     = normalize_slat(_g2_x0v)
        flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
        _g2_color, _g2_leaf = decode_and_render(pipeline, _g2_sv, renderer, diag=False, device=DEVICE)
        flow_model.to(DEVICE)
        _g2_gt, _g2_gm = load_gt(75)
        _, _, _g2_loss = masked_loss(_g2_color, _g2_gt, render_mask, _g2_gm, lpips_fn)
        (_g2_loss * loss_scale).backward()
        _g2_grad = _g2_leaf.grad
        print(f'  slat_leaf.grad: max={_g2_grad.abs().max():.3e}  mean={_g2_grad.abs().mean():.3e}')
        _g2_x0r = denoise_last_step(flow_model, _g2_prefix, _g2_cond, lora, require_grad=True)
        _g2_sr   = normalize_slat(_g2_x0r)
        torch.autograd.backward(_g2_sr.feats, _g2_grad)
        for p in lora.parameters():
            if p.grad is not None:
                p.grad.div_(loss_scale)
        b_gnorms = b_grad_norms(lora)
        n_b      = N_ACTIVE * 2   # q + kv per active block
        n_zero   = sum(1 for g in b_gnorms if g == 0.0)
        print(f'  B.grad norms ({len(b_gnorms)}/{n_b} have grad): '
              f'min={min(b_gnorms):.3e}  max={max(b_gnorms):.3e}')
        if n_zero > 0:
            print(f'  [GATE 2] WARNING: {n_zero}/{n_b} B matrices have exactly-zero grad '
                  f'(expected for early blocks in fp16; training dominated by later blocks)')
        assert len(b_gnorms) == n_b, f'GATE 2 FAILED: only {len(b_gnorms)}/{n_b} B matrices received grad'
        print('[GATE 2] PASSED\n')
        optimizer.zero_grad(set_to_none=True)
        del _g2_cond, _g2_noise, _g2_prefix, _g2_x0v, _g2_sv, _g2_color, _g2_leaf
        del _g2_gt, _g2_gm, _g2_loss, _g2_x0r, _g2_sr, _g2_grad
        gc.collect(); torch.cuda.empty_cache()

    # ── training ──────────────────────────────────────────────────────────────
    raw_cache  = {}
    frame_list = [1, 75, 150] if args.smoke else TRAIN

    print(f'[TRAIN] epochs {start_epoch}→{EPOCHS}  '
          f'frames/epoch={len(frame_list)}  LOSS_SCALE={loss_scale}')

    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep = time.time()
        epoch_mse = epoch_lp = 0.0
        n_steps = 0
        train_frames = frame_list if args.smoke else random.sample(TRAIN, len(TRAIN))

        for frame_i in train_frames:
            cond_gl  = raw_tokens[frame_i].unsqueeze(0).to(DEVICE)
            noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
            optimizer.zero_grad(set_to_none=True)

            x_prefix = denoise_prefix_nograd(flow_model, noise_sp, cond_gl, lora)
            x0_val   = denoise_last_step(flow_model, x_prefix, cond_gl, lora, require_grad=False)
            slat_v   = normalize_slat(x0_val)

            flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()

            try:
                color, leaf = decode_and_render(pipeline, slat_v, renderer, diag=False, device=DEVICE)
                gc.collect(); torch.cuda.empty_cache()
                gt, gt_mask = load_gt(frame_i)
                mse_t, lp_t, total_loss = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)
                (total_loss * loss_scale).backward()
            finally:
                flow_model.to(DEVICE)

            if leaf.grad is None:
                print(f'  e{epoch:02d} f{frame_i:03d}  SKIP: no grad on slat_leaf')
                optimizer.zero_grad(set_to_none=True)
                del color, leaf, gt, gt_mask, mse_t, lp_t, total_loss, x0_val, slat_v, noise_sp, x_prefix, cond_gl
                gc.collect(); torch.cuda.empty_cache()
                continue

            grad_slat = leaf.grad.detach()
            x0_raw    = denoise_last_step(flow_model, x_prefix, cond_gl, lora, require_grad=True)
            slat_r    = normalize_slat(x0_raw)
            torch.autograd.backward(slat_r.feats, grad_slat)

            for p in lora.parameters():
                if p.grad is not None:
                    p.grad.div_(loss_scale)

            has_bad = any(not torch.isfinite(p.grad).all()
                          for p in lora.parameters() if p.grad is not None)
            if has_bad:
                prev = loss_scale
                loss_scale = max(loss_scale / 2.0, 1.0)
                print(f'  e{epoch:02d} f{frame_i:03d}  inf/nan → skip  '
                      f'loss_scale {prev:.0f}→{loss_scale:.0f}')
                optimizer.zero_grad(set_to_none=True)
                del color, leaf, gt, gt_mask, mse_t, lp_t, total_loss
                del x0_val, slat_v, x0_raw, slat_r, grad_slat, noise_sp, x_prefix, cond_gl
                gc.collect(); torch.cuda.empty_cache()
                continue

            nn.utils.clip_grad_norm_(lora.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_mse += mse_t.item()
            epoch_lp  += lp_t.item()
            n_steps   += 1

            if n_steps <= 3 or frame_i % 30 == 0:
                print(f'  e{epoch:02d} f{frame_i:03d}  '
                      f'mse={mse_t.item():.5f}  lpips={lp_t.item():.5f}  '
                      f'total={total_loss.item():.5f}')

            del color, leaf, gt, gt_mask, mse_t, lp_t, total_loss
            del x0_val, slat_v, x0_raw, slat_r, grad_slat, noise_sp, x_prefix, cond_gl
            gc.collect(); torch.cuda.empty_cache()

        # ── epoch stats ───────────────────────────────────────────────────────
        avg_mse = epoch_mse / max(n_steps, 1)
        avg_lp  = epoch_lp  / max(n_steps, 1)
        avg_tot = avg_mse + W_LPIPS * avg_lp

        norms = registry_norms(lora, w_q_per_block)
        B_norms   = [v['B_norm']   for v in norms.values()]
        BA_norms  = [v['BA_norm']  for v in norms.values()]
        BA_ratios = [v['BA_ratio'] for v in norms.values()]

        held_psnr, held_psnr_std, held_lpips, held_ssim, held_ssim_std, held_res = \
            evaluate_held_out(pipeline, flow_model, lora, raw_tokens, coords,
                              fixed_noise_feats, renderer, render_mask, lpips_fn)

        is_best = held_psnr > best_psnr
        if is_best:
            best_psnr = held_psnr

        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        elapsed = time.time() - t_ep
        print(f'\n[EPOCH {epoch:02d}/{EPOCHS}]  '
              f'loss={avg_tot:.5f}  (mse={avg_mse:.5f}  lpips×{W_LPIPS}={W_LPIPS*avg_lp:.5f})')
        print(f'  ||B||   mean={np.mean(B_norms):.4f}  '
              f'min={np.min(B_norms):.4f}  max={np.max(B_norms):.4f}')
        print(f'  BA/Wq   mean={np.mean(BA_ratios):.4f}  '
              f'min={np.min(BA_ratios):.4f}  max={np.max(BA_ratios):.4f}')
        b_gn = b_grad_norms(lora)
        if b_gn:
            print(f'  B.grad  min={min(b_gn):.3e}  max={max(b_gn):.3e}')
        print(f'  held PSNR={held_psnr:.3f}±{held_psnr_std:.3f} dB  '
              f'SSIM={held_ssim:.4f}  LPIPS={held_lpips:.4f}  '
              f'GPU={gpu_mem_gb:.1f}GB  loss_scale={loss_scale:.0f}  '
              f'{"BEST ✓" if is_best else ""}  time={elapsed:.1f}s\n')

        per_block_ratios = {k: v['BA_ratio'] for k, v in norms.items()}

        row = {
            'epoch': epoch,
            'loss_total': avg_tot, 'loss_mse': avg_mse, 'loss_lpips': avg_lp,
            'held_psnr': held_psnr, 'held_psnr_std': held_psnr_std,
            'held_ssim': held_ssim, 'held_ssim_std': held_ssim_std,
            'held_lpips': held_lpips, 'held_lpips_std': held_res['lpips_std'],
            'held_per_frame': held_res['per_frame'],
            'B_norm_mean': float(np.mean(B_norms)),
            'BA_norm_mean': float(np.mean(BA_norms)),
            'BA_ratio_mean': float(np.mean(BA_ratios)),
            'BA_ratio_per_block': per_block_ratios,
            'loss_scale': loss_scale,
            'n_steps': n_steps,
            'peak_gpu_gb': round(gpu_mem_gb, 3),
            'wall_secs': elapsed,
        }
        history.append(row)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)

        if not args.smoke:
            ckpt_data = {
                'run_id': RUN_ID, 'epoch': epoch,
                'lora_state': lora.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr, 'held_psnr': held_psnr,
                'loss_scale': loss_scale,
            }
            torch.save(ckpt_data, _CKPT / f'lora_e{epoch:03d}.pt')
            if is_best:
                torch.save(ckpt_data, _CKPT / 'lora_best.pt')
                print(f'  [CKPT] new best → lora_best.pt')

        if epoch % DIAG_EVERY == 0 or epoch == EPOCHS:
            save_diagnostics(pipeline, flow_model, lora, raw_tokens, coords,
                             fixed_noise_feats, renderer, epoch, raw_cache)
            save_curves(history, _LABEL)
            gc.collect(); torch.cuda.empty_cache()

    # ── Final 150-frame eval ─────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('[FINAL EVAL] All 150 frames — post-training quality audit')
    print('=' * 72)
    ALL_FRAMES = list(range(1, N_FRAMES + 1))
    final_res  = evaluate_frames(pipeline, flow_model, lora, raw_tokens, coords,
                                 fixed_noise_feats, renderer, render_mask, lpips_fn,
                                 ALL_FRAMES)

    held_mask  = [f in set(HELD_OUT) for f in ALL_FRAMES]
    train_mask = [f in set(TRAIN)    for f in ALL_FRAMES]

    def _split(key):
        held  = [r[key] for r, h in zip(final_res['per_frame'], held_mask)  if h]
        train = [r[key] for r, t in zip(final_res['per_frame'], train_mask) if t]
        return held, train

    hp, tp = _split('psnr'); hs, ts = _split('ssim'); hl, tl = _split('lpips')
    final_summary = {
        'placement': args.placement, 'rank': args.rank, 'run_id': RUN_ID,
        'all':   {k: final_res[k] for k in
                  ('psnr_mean','psnr_std','ssim_mean','ssim_std','lpips_mean','lpips_std')},
        'held':  dict(psnr_mean=float(np.mean(hp)),  psnr_std=float(np.std(hp)),
                      ssim_mean=float(np.mean(hs)),  ssim_std=float(np.std(hs)),
                      lpips_mean=float(np.mean(hl)), lpips_std=float(np.std(hl))),
        'train': dict(psnr_mean=float(np.mean(tp)),  psnr_std=float(np.std(tp)),
                      ssim_mean=float(np.mean(ts)),  ssim_std=float(np.std(ts)),
                      lpips_mean=float(np.mean(tl)), lpips_std=float(np.std(tl))),
        'per_frame': final_res['per_frame'],
    }
    json.dump(final_summary, open(_OUT / 'final_eval.json', 'w'), indent=2)
    print(f'  All 150: PSNR={final_summary["all"]["psnr_mean"]:.3f}±'
          f'{final_summary["all"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["all"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["all"]["lpips_mean"]:.4f}')
    print(f'  Held:    PSNR={final_summary["held"]["psnr_mean"]:.3f}±'
          f'{final_summary["held"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["held"]["ssim_mean"]:.4f}')
    print(f'  Train:   PSNR={final_summary["train"]["psnr_mean"]:.3f}±'
          f'{final_summary["train"]["psnr_std"]:.3f} dB')
    print(f'  → final_eval.json saved')

    print(f'\n[DONE] {EPOCHS} epochs complete.')
    print(f'  placement  : {args.placement}  blocks={ACTIVE_BLOCKS}  rank={args.rank}')
    print(f'  best PSNR  : {best_psnr:.3f} dB')
    print(f'  output     : {_OUT}')


if __name__ == '__main__':
    main()
