"""
Run 01 — Single-path LoRA on raw TRELLIS.

Question: does standard LoRA on frozen TRELLIS improve per-frame texture quality?

No MCFM, no blending, no dual-path, no alpha scalar.
Frame t's DINOv2 tokens → LoRA-patched TRELLIS → SLaT decode → render → loss vs GT.

All five historical bugs fixed (see spec):
  FIX 1: A init kaiming_uniform
  FIX 2: scaling = lora_alpha / rank = 1.0
  FIX 3: LoRA forward in fp32
  FIX 4: LOSS_SCALE kept through flow backward, unscaled only before optimizer
  FIX 5: Adam eps=1e-16

Output: lora_experiments/runs/rung1_single_path_r4_s6/
  lora_ckpts/lora_e{N:03d}.pt
  diag_renders/e{N:03d}/strip_f{F:04d}.png
  loss_history.json
  train.log
"""

import sys, os, argparse as _ap, hashlib
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
_OUT_DIR = _HERE / 'runs' / 'rung1_single_path_r4_s6'
_CKPT    = _OUT_DIR / 'lora_ckpts'
_DIAG    = _OUT_DIR / 'diag_renders'

_OUT_DIR.mkdir(parents=True, exist_ok=True)
_CKPT.mkdir(exist_ok=True)
_DIAG.mkdir(exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(_OUT_DIR / 'train.log')
sys.stderr = sys.stdout


# ── nvdiffrast arch guard (same pattern as step07d) ──────────────────────────
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

import json, math, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from contextlib import contextmanager
from skimage.metrics import structural_similarity as _ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step6_5_lora.lora_v2_fixed import (
    build_lora_blocks, freeze_trellis, gate0_verify,
    trainable_params, count_trainable,
)
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, decode_and_render, RENDER_RES,
)

# ── Constants ─────────────────────────────────────────────────────────────────
GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH  = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
N_FRAMES    = 150
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
RESCALE_T   = 3.0
LORA_RANK   = 4
LR          = 1e-4
LOSS_SCALE0 = 4096.0
GRAD_CLIP   = 1.0
W_LPIPS     = 0.1
N_EPOCHS    = 20

HELD_OUT = list(range(5, 151, 10))    # [5, 15, 25, ..., 145]  15 frames, never trained on
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]   # 135 frames

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]


# ── Single-path attention context manager ─────────────────────────────────────

_ATTN_CHUNK = 256

def _attn_chunked(q, k, v, scale, chunk=_ATTN_CHUNK):
    """Chunked scaled dot-product attention. q:(N,H,hd) k/v:(T,H,hd) -> (N,H,hd)."""
    outs = []
    for s in range(0, q.shape[0], chunk):
        e   = min(s + chunk, q.shape[0])
        qc  = q[s:e].float()
        A   = torch.einsum('nhd,mhd->nhm', qc, k.float()) * scale
        A   = torch.softmax(A, dim=-1)
        out = torch.einsum('nhm,mhd->nhd', A, v.float())
        outs.append(out)
    return torch.cat(outs, dim=0)


def _single_path_fwd(module, x, context, lora_block):
    """
    Single-path LoRA cross-attention.

    q  = to_q(x_feats)  + lora_q(x_feats)      <- same input space as to_q
    kv = to_kv(cond)    + lora_kv(cond)         <- same input space as to_kv
    out = to_out(attn(q, k, v))

    No alpha, no K_pooled, no enhance_bias.
    At B=0: output == raw TRELLIS (GATE 1).
    """
    ch  = module.channels
    H   = module.num_heads
    hd  = ch // H
    scale = hd ** -0.5
    wdt = module.to_q.weight.dtype

    x_feats = x.feats.to(wdt)                                      # (N, ch)
    ctx_2d  = (context.squeeze(0) if context.dim() == 3
               else context).to(wdt)                               # (T, cond_ch)

    # frozen projections — no grad here
    with torch.no_grad():
        q_base  = module.to_q(x_feats)                             # (N, ch)
        kv_base = module.to_kv(ctx_2d)                             # (T, 2*ch)

    # LoRA deltas — grad flows through these when B≠0
    delta_q  = lora_block.lora_q(x_feats)                         # (N, ch)
    delta_kv = lora_block.lora_kv(ctx_2d)                         # (T, 2*ch)

    q  = (q_base  + delta_q ).reshape(-1, H, hd)                   # (N, H, hd)
    kv = (kv_base + delta_kv).reshape(-1, 2, H, hd)                # (T, 2, H, hd)
    k, v = kv[:, 0], kv[:, 1]

    out = _attn_chunked(q, k, v, scale)                            # (N, H, hd)
    out = out.to(wdt).reshape(-1, ch)                              # (N, ch)
    out = module.to_out(out)                                       # (N, ch)
    return x.replace(out.to(x.feats.dtype))


@contextmanager
def single_path_ctx(flow_model, lora_blocks):
    """Patch all 24 SLAT-flow cross-attn blocks with single-path LoRA forward."""
    saved    = {}
    lora_idx = 0
    for i, block in enumerate(flow_model.blocks):
        if not hasattr(block, 'cross_attn'):
            continue
        ca       = block.cross_attn
        saved[i] = ca.forward
        lb       = lora_blocks[lora_idx]
        lora_idx += 1

        def _make(mod, lb_):
            def _fwd(x, context=None):
                return _single_path_fwd(mod, x, context, lb_)
            return _fwd

        ca.forward = _make(ca, lb)

    assert lora_idx == 24, f'Expected 24 cross-attn blocks, found {lora_idx}'
    try:
        yield
    finally:
        for i, block in enumerate(flow_model.blocks):
            if i in saved:
                block.cross_attn.forward = saved[i]


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)   # (n_tok, 1024)


def load_gt(frame_idx: int):
    """Returns (gt: Tensor(3,H,W) in [0,1], gt_mask: Tensor(H,W) bool)."""
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt  = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = (gt < 0.99).any(dim=0)   # any non-white pixel = object
    return gt, gt_mask


def load_render_mask() -> torch.Tensor:
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)   # (H, W) bool


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Full-frame SSIM (3,H,W) in [0,1]. skimage expects (H,W,C)."""
    p = pred.permute(1, 2, 0).cpu().numpy()
    g = gt.permute(1, 2, 0).cpu().numpy()
    return float(_ssim(p, g, data_range=1.0, channel_axis=2))


def masked_loss(rendered, gt, render_mask, gt_mask, lpips_fn):
    m   = (render_mask | gt_mask).float()                          # union
    mse = ((rendered - gt) ** 2 * m).sum() / (m.sum() * 3 + 1e-8)
    r   = rendered * m + (1 - m)                                   # composite onto white
    g   = gt       * m + (1 - m)
    lp  = lpips_fn(r.unsqueeze(0) * 2 - 1,
                   g.unsqueeze(0) * 2 - 1).mean()
    return mse, lp, mse + W_LPIPS * lp


# ── Denoising helpers ─────────────────────────────────────────────────────────

def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, lora_blocks):
    """Steps 0..23 (T_PAIRS[:-1]), no grad."""
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with single_path_ctx(flow_model, lora_blocks):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, lora_blocks, require_grad: bool):
    """Final Euler step (T_PAIRS[-1])."""
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    if require_grad:
        with single_path_ctx(flow_model, lora_blocks):
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with single_path_ctx(flow_model, lora_blocks):
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl, lora_blocks):
    """All 25 steps, no grad. Used for eval and diagnostics."""
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with single_path_ctx(flow_model, lora_blocks):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


# ── Norms ─────────────────────────────────────────────────────────────────────

def block_norms(lora_blocks, w_norms_per_layer=None):
    """
    Per-layer norms. w_norms_per_layer: list of (w_q_norm, w_kv_norm) per block.
    eff_ratio uses each layer's own frozen weight norm, not a single block-0 ref.
    """
    A_norms, B_norms, BA_norms, eff_ratios = [], [], [], []
    per_block_ratios = {}   # block_idx -> {q: ratio, kv: ratio}
    for b_idx, blk in enumerate(lora_blocks):
        wq_norm  = w_norms_per_layer[b_idx][0] if w_norms_per_layer else 1.0
        wkv_norm = w_norms_per_layer[b_idx][1] if w_norms_per_layer else 1.0
        ratios_blk = {}
        for name, layer, wnorm in [('q', blk.lora_q, wq_norm), ('kv', blk.lora_kv, wkv_norm)]:
            A  = layer.A.float(); B = layer.B.float()
            ba = (B @ A).norm().item()
            A_norms.append(A.norm().item())
            B_norms.append(B.norm().item())
            BA_norms.append(ba)
            ratio = ba / wnorm
            eff_ratios.append(ratio)
            ratios_blk[name] = ratio
        per_block_ratios[b_idx] = ratios_blk
    return A_norms, B_norms, BA_norms, eff_ratios, per_block_ratios


def b_grad_norms(lora_blocks):
    norms = []
    for blk in lora_blocks:
        for layer in (blk.lora_q, blk.lora_kv):
            g = layer.B.grad
            if g is not None:
                norms.append(g.norm().item())
    return norms


# ── Diagnostics ───────────────────────────────────────────────────────────────

def make_strip(panels, cell=320, label_h=28, font=None):
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
        try:   tw = draw.textbbox((0, 0), lbl, font=font)[2]
        except: tw = len(lbl) * 7
        draw.text((col * cell + (cell - tw) // 2, 6), lbl, fill=(210, 210, 210), font=font)
    return canvas


def save_diagnostics(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                     fixed_noise_feats, renderer, epoch, raw_cache, font):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    flow_model.eval()
    for fi in DIAG_FRAMES:
        cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl, lora_blocks)
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render   = color.detach().clamp(0, 1)

        if fi not in raw_cache:
            # raw TRELLIS: zero-B lora_blocks gives identical output to no-LoRA (GATE 1)
            # We render once with current blocks — at e0 this IS raw TRELLIS
            raw_cache[fi] = render.clone()

        strip = make_strip([
            (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
            (raw_cache[fi],                          'TRELLIS baseline'),
            (render,                                 f'LoRA e{epoch:03d}'),
        ], font=font)
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'render_f{fi:04d}.png')
        del render, slat, color
        gc.collect(); torch.cuda.empty_cache()
    flow_model.train()


def save_loss_curve(history):
    if len(history) < 2:
        return
    ep    = [r['epoch']     for r in history]
    loss  = [r['loss_total'] for r in history]
    mse_  = [r['loss_mse']  for r in history]
    lp_   = [r['loss_lpips'] for r in history]
    psnr_ = [r['held_psnr'] for r in history]
    bn    = [r['B_norm_mean'] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Run 01 — Single-path LoRA', fontsize=11)
    axes[0].plot(ep, mse_, 'o-', color='#e06c75', lw=2, ms=4, label='MSE')
    axes[0].plot(ep, lp_,  's-', color='#d19a66', lw=2, ms=4, label='LPIPS×0.1')
    axes[0].plot(ep, loss, '^-', color='#c678dd', lw=2, ms=4, label='total')
    axes[0].set_title('Train loss'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, psnr_, 'o-', color='#98c379', lw=2, ms=4)
    axes[1].set_title('Held-out masked PSNR (dB)'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, bn, 'o-', color='#61afef', lw=2, ms=4)
    axes[2].set_title('Mean ||B|| across 48 matrices'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── Held-out evaluation ───────────────────────────────────────────────────────

def evaluate_frames(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                    fixed_noise_feats, renderer, render_mask, lpips_fn, frame_list):
    """Evaluate a list of frames. Returns aggregate stats + per-frame list."""
    flow_model.eval()
    per_frame = []
    for fi in frame_list:
        cond_gl  = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat     = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl, lora_blocks)
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
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
    psnrs = [r['psnr'] for r in per_frame]
    ssims = [r['ssim'] for r in per_frame]
    lpips = [r['lpips'] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'lpips_mean': float(np.mean(lpips)), 'lpips_std': float(np.std(lpips)),
        'per_frame': per_frame,
    }


def evaluate_held_out(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                      fixed_noise_feats, renderer, render_mask, lpips_fn, epoch):
    res = evaluate_frames(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                          fixed_noise_feats, renderer, render_mask, lpips_fn, HELD_OUT)
    return res['psnr_mean'], res['psnr_std'], res['lpips_mean'], res['ssim_mean'], res


# ── Checkpoint ────────────────────────────────────────────────────────────────

def find_latest_ckpt():
    ckpts = sorted(_CKPT.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=N_EPOCHS)
    parser.add_argument('--loss_scale', type=float, default=LOSS_SCALE0)
    parser.add_argument('--smoke',      action='store_true')
    parser.add_argument('--diag_every', type=int,   default=1)
    args = parser.parse_args()

    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = args.loss_scale
    DIAG_EVERY = args.diag_every

    # config.json + MD5 hash (spec A.10)
    _cfg = dict(run='run01', rank=LORA_RANK, layers=['to_q','to_kv'], n_blocks=24,
                lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
                epochs=EPOCHS, held_out=HELD_OUT, struct_seed=STRUCT_SEED,
                fixed_seed=FIXED_SEED, grad_clip=GRAD_CLIP)
    _run_id = hashlib.md5(json.dumps(_cfg, sort_keys=True).encode()).hexdigest()[:8]
    _cfg['run_id'] = _run_id
    with open(_OUT_DIR / 'config.json', 'w') as _f:
        json.dump(_cfg, _f, indent=2)

    print('=' * 72)
    print('Run 01 — Single-path LoRA, raw TRELLIS, rank=4, to_q/to_kv, 24 blocks')
    print('=' * 72)
    print(f'  run_id     : {_run_id}')
    print(f'  epochs     : {EPOCHS}')
    print(f'  loss_scale : {loss_scale}')
    print(f'  held_out   : {HELD_OUT}  ({len(HELD_OUT)} frames)')
    print(f'  train      : {len(TRAIN)} frames')
    print(f'  output     : {_OUT_DIR}')

    # ── Gate: verify train/held splits are disjoint ───────────────────────────
    assert len(set(HELD_OUT) & set(TRAIN)) == 0, 'HELD_OUT / TRAIN overlap!'
    assert len(HELD_OUT) == 15 and len(TRAIN) == 135

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print(f'\n[LOAD] Loading pipeline from {PRETRAINED}...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Structure (fixed seed) ────────────────────────────────────────────────
    # Sample structure BEFORE offloading: sample_sparse_structure needs the
    # sparse-structure models on GPU, and pipeline.device is derived from the
    # first model's device — offloading it first makes get_cond move the image
    # to CPU while DINOv2 weights stay on CUDA (device mismatch crash).
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

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print('\n[LORA] Building LoRA blocks (rank=4, lora_alpha=4 → scaling=1.0)...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, lora_alpha=LORA_RANK).to(DEVICE)
    gate0_verify(lora_blocks)

    n_train = count_trainable(lora_blocks)
    print(f'\n[GATE 0] trainable params: {n_train:,}')
    assert n_train == 491_520, f'param gate FAILED: {n_train} != 491_520'
    assert not any(p.requires_grad for p in flow_model.parameters()), \
        'frozen TRELLIS has requires_grad=True!'
    print('[GATE 0] PASSED\n')

    # Per-block, per-layer W norms for ||B@A||/||W|| ratio (spec A.10)
    w_norms_per_layer = []
    for blk in flow_model.blocks:
        if hasattr(blk, 'cross_attn'):
            ca = blk.cross_attn
            w_norms_per_layer.append((
                ca.to_q.weight.float().norm().item(),
                ca.to_kv.weight.float().norm().item(),
            ))
    print(f'  ||W_q|| block 0 = {w_norms_per_layer[0][0]:.4f}  '
          f'||W_kv|| block 0 = {w_norms_per_layer[0][1]:.4f}')

    # ── Optimizer + resume ────────────────────────────────────────────────────
    optimizer   = torch.optim.Adam(lora_blocks.parameters(), lr=LR,
                                   eps=1e-16, weight_decay=0.0)
    start_epoch = 1
    history     = []
    best_psnr   = -float('inf')

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None
    if latest_ckpt:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        assert ckpt.get('run') == 'run01', f'wrong checkpoint: run={ckpt.get("run")}'
        lora_blocks.load_state_dict(ckpt['lora_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -float('inf'))
        hist_path   = _OUT_DIR / 'loss_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
        print(f'  resumed from epoch {ckpt["epoch"]}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] FIXED_SEED={FIXED_SEED}')
    torch.manual_seed(FIXED_SEED)
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

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    print('\n[LPIPS] Loading AlexNet...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ── Render mask ───────────────────────────────────────────────────────────
    render_mask = load_render_mask()
    print(f'[MASK] {render_mask.sum().item()} masked pixels '
          f'({100*render_mask.float().mean():.1f}%)')

    renderer = make_renderer()
    pipeline.models['slat_decoder_mesh'].to(DEVICE)

    # ── GATE 1: identity at init (fresh runs only) ────────────────────────────
    # On resume, B is trained (non-zero) by design, so the B=0 identity check no
    # longer holds — skip it rather than crash on the assertion.
    if resumed:
        print('\n[GATE 1] SKIPPED — resuming from checkpoint (B ≠ 0 by design)\n')
    else:
        print('\n[GATE 1] Identity check (B=0 → LoRA output == raw TRELLIS)...')
        _g1_cond = raw_tokens[75].unsqueeze(0).to(DEVICE)
        _g1_ns   = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        with torch.no_grad():
            # raw TRELLIS (no context manager)
            _g1_raw = _g1_ns
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                v = flow_model(_g1_raw, t_ten, _g1_cond)
                _g1_raw = _g1_raw.replace(_g1_raw.feats - (t - t_prev) * v.feats)
            # single-path LoRA (B=0)
            _g1_lora = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                with single_path_ctx(flow_model, lora_blocks):
                    v = flow_model(_g1_lora, t_ten, _g1_cond)
                _g1_lora = _g1_lora.replace(_g1_lora.feats - (t - t_prev) * v.feats)
        diff = (_g1_raw.feats - _g1_lora.feats).abs().max().item()
        print(f'  max |raw - lora(B=0)| = {diff:.3e}')
        assert diff < 1e-2, f'GATE 1 FAILED: max diff = {diff:.3e}'
        print('[GATE 1] PASSED\n')
        del _g1_raw, _g1_lora, _g1_cond
        gc.collect(); torch.cuda.empty_cache()

    # ── GATE 2: gradient flow check (frame 75, one backward) ──────────────────
    print('[GATE 2] Gradient flow check (frame 75, one backward)...')
    flow_model.train()
    _g2_cond    = raw_tokens[75].unsqueeze(0).to(DEVICE)
    _g2_noise   = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    _g2_prefix  = denoise_prefix_nograd(flow_model, _g2_noise, _g2_cond, lora_blocks)
    _g2_x0_val  = denoise_last_step(flow_model, _g2_prefix, _g2_cond, lora_blocks, require_grad=False)
    _g2_slat_v  = normalize_slat(_g2_x0_val)
    _g2_color, _g2_leaf = decode_and_render(pipeline, _g2_slat_v, renderer, diag=False, device=DEVICE)
    _g2_gt, _g2_gm      = load_gt(75)
    _, _, _g2_loss = masked_loss(_g2_color, _g2_gt, render_mask, _g2_gm, lpips_fn)
    (_g2_loss * loss_scale).backward()
    _g2_grad = _g2_leaf.grad
    print(f'  slat_leaf.grad: max={_g2_grad.abs().max():.3e}  mean={_g2_grad.abs().mean():.3e}')
    _g2_x0_raw = denoise_last_step(flow_model, _g2_prefix, _g2_cond, lora_blocks, require_grad=True)
    _g2_slat_r  = normalize_slat(_g2_x0_raw)
    torch.autograd.backward(_g2_slat_r.feats, _g2_grad)
    # unscale
    for p in lora_blocks.parameters():
        if p.grad is not None:
            p.grad.div_(loss_scale)
    b_gnorms = b_grad_norms(lora_blocks)
    print(f'  B.grad norms ({len(b_gnorms)}/48 nonzero): '
          f'min={min(b_gnorms):.3e}  max={max(b_gnorms):.3e}')
    assert len(b_gnorms) == 48, f'Only {len(b_gnorms)}/48 B matrices have grad'
    assert all(g > 0 for g in b_gnorms), 'Some B matrices have zero gradient'
    print('[GATE 2] PASSED\n')
    optimizer.zero_grad(set_to_none=True)
    del _g2_cond, _g2_noise, _g2_prefix, _g2_x0_val, _g2_slat_v, _g2_color, _g2_leaf
    del _g2_gt, _g2_gm, _g2_loss, _g2_x0_raw, _g2_slat_r, _g2_grad
    gc.collect(); torch.cuda.empty_cache()

    # ── Training ──────────────────────────────────────────────────────────────
    raw_cache  = {}   # stores baseline render for diag strips
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

            x_prefix = denoise_prefix_nograd(flow_model, noise_sp, cond_gl, lora_blocks)

            # no-grad pass for decode/render
            x0_val  = denoise_last_step(flow_model, x_prefix, cond_gl, lora_blocks, require_grad=False)
            slat_v  = normalize_slat(x0_val)

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
                del color, leaf, gt, gt_mask, mse_t, lp_t, total_loss, x0_val, slat_v
                del noise_sp, x_prefix, cond_gl
                gc.collect(); torch.cuda.empty_cache()
                continue

            grad_slat = leaf.grad.detach()   # still at loss_scale amplitude

            # grad pass through flow model
            x0_raw = denoise_last_step(flow_model, x_prefix, cond_gl, lora_blocks, require_grad=True)
            slat_r  = normalize_slat(x0_raw)
            torch.autograd.backward(slat_r.feats, grad_slat)

            # unscale before optimizer
            for p in lora_blocks.parameters():
                if p.grad is not None:
                    p.grad.div_(loss_scale)

            has_bad = any(not torch.isfinite(p.grad).all()
                          for p in lora_blocks.parameters() if p.grad is not None)
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

            torch.nn.utils.clip_grad_norm_(lora_blocks.parameters(), GRAD_CLIP)
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

        # ── Epoch stats ───────────────────────────────────────────────────────
        avg_mse = epoch_mse / max(n_steps, 1)
        avg_lp  = epoch_lp  / max(n_steps, 1)
        avg_tot = avg_mse + W_LPIPS * avg_lp

        A_norms, B_norms, BA_norms, eff_ratios, per_block_ratios = block_norms(lora_blocks, w_norms_per_layer)

        # held-out eval
        held_psnr, held_psnr_std, held_lpips, held_ssim, held_res = evaluate_held_out(
            pipeline, flow_model, lora_blocks, raw_tokens, coords,
            fixed_noise_feats, renderer, render_mask, lpips_fn, epoch)

        is_best = held_psnr > best_psnr
        if is_best:
            best_psnr = held_psnr

        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        elapsed = time.time() - t_ep

        b_gn = b_grad_norms(lora_blocks)

        print(f'\n[EPOCH {epoch:02d}]  loss={avg_tot:.5f}  '
              f'(mse={avg_mse:.5f}  lpips×{W_LPIPS}={W_LPIPS*avg_lp:.5f})')
        print(f'  ||A|| mean={np.mean(A_norms):.4f}  '
              f'min={np.min(A_norms):.4f}  max={np.max(A_norms):.4f}')
        print(f'  ||B|| mean={np.mean(B_norms):.4f}  '
              f'min={np.min(B_norms):.4f}  max={np.max(B_norms):.4f}')
        print(f'  ||B@A|| mean={np.mean(BA_norms):.4f}  '
              f'||B@A||/||W|| mean={np.mean(eff_ratios):.4f}  '
              f'min={np.min(eff_ratios):.4f}  max={np.max(eff_ratios):.4f}')
        if b_gn:
            print(f'  B.grad norms: min={min(b_gn):.3e}  max={max(b_gn):.3e}')
        print(f'  held PSNR={held_psnr:.3f}±{held_psnr_std:.3f} dB  '
              f'SSIM={held_ssim:.4f}  LPIPS={held_lpips:.4f}  '
              f'loss_scale={loss_scale:.0f}  GPU={gpu_mem_gb:.1f}GB  '
              f'{"BEST ✓" if is_best else ""}  time={elapsed:.1f}s\n')

        row = {
            'epoch': epoch, 'loss_total': avg_tot,
            'loss_mse': avg_mse, 'loss_lpips': avg_lp,
            'held_psnr': held_psnr, 'held_psnr_std': held_psnr_std,
            'held_ssim': held_ssim, 'held_ssim_std': held_res['ssim_std'],
            'held_lpips': held_lpips, 'held_lpips_std': held_res['lpips_std'],
            'held_per_frame': held_res['per_frame'],
            'A_norm_mean': float(np.mean(A_norms)),
            'A_norm_min':  float(np.min(A_norms)),
            'A_norm_max':  float(np.max(A_norms)),
            'B_norm_mean': float(np.mean(B_norms)),
            'B_norm_min':  float(np.min(B_norms)),
            'B_norm_max':  float(np.max(B_norms)),
            'BA_norm_mean': float(np.mean(BA_norms)),
            'eff_ratio_mean': float(np.mean(eff_ratios)),
            'eff_ratio_min':  float(np.min(eff_ratios)),
            'eff_ratio_max':  float(np.max(eff_ratios)),
            'per_block_ratios': {str(k): v for k, v in per_block_ratios.items()},
            'B_grad_min': float(min(b_gn)) if b_gn else None,
            'B_grad_max': float(max(b_gn)) if b_gn else None,
            'loss_scale': loss_scale,
            'n_steps': n_steps,
            'peak_gpu_gb': round(gpu_mem_gb, 3),
            'wall_secs': elapsed,
        }
        history.append(row)
        with open(_OUT_DIR / 'loss_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if not args.smoke:
            ckpt_data = {
                'run': 'run01', 'epoch': epoch,
                'lora_state': lora_blocks.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'held_psnr': held_psnr,
            }
            torch.save(ckpt_data, _CKPT / f'lora_e{epoch:03d}.pt')
            if is_best:
                torch.save(ckpt_data, _CKPT / 'lora_best.pt')
                print(f'  [CKPT] new best → lora_best.pt')

        if epoch % DIAG_EVERY == 0 or epoch == EPOCHS:
            save_diagnostics(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                             fixed_noise_feats, renderer, epoch, raw_cache, font=None)
            save_loss_curve(history)
            gc.collect(); torch.cuda.empty_cache()

    # ── Free figure: ||B@A||/||W|| vs block index (spec Rung 1) ─────────────────
    # Uses the last epoch's per_block_ratios from history
    if history and 'per_block_ratios' in history[-1]:
        last_ratios = history[-1]['per_block_ratios']
        blocks  = sorted(int(k) for k in last_ratios.keys())
        q_ratio = [last_ratios[str(b)]['q']  for b in blocks]
        kv_ratio= [last_ratios[str(b)]['kv'] for b in blocks]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(blocks, q_ratio,  'o-', color='#98c379', lw=2, ms=5, label='lora_q / ||W_q||')
        ax.plot(blocks, kv_ratio, 's-', color='#61afef', lw=2, ms=5, label='lora_kv / ||W_kv||')
        ax.axhline(0.1, color='#e06c75', lw=1, ls='--', label='refinement ceiling (0.1)')
        ax.axhline(0.01, color='#d19a66', lw=1, ls=':', label='refinement floor (0.01)')
        ax.set_xlabel('Block index'); ax.set_ylabel('||B@A|| / ||W||')
        ax.set_title('Rung 1 — Per-block LoRA correction magnitude (epoch 20)\n'
                     'Concentration in early blocks → predicts Rung 2 outcome')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(_DIAG / 'per_block_BA_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  per-block figure   : {_DIAG}/per_block_BA_ratio.png')

    # ── Final full 150-frame evaluation ──────────────────────────────────────────
    print('\n' + '=' * 68)
    print('[FINAL EVAL] All 150 frames — post-training quality audit')
    print('=' * 68)
    ALL_FRAMES = list(range(1, N_FRAMES + 1))
    final_res  = evaluate_frames(pipeline, flow_model, lora_blocks, raw_tokens, coords,
                                 fixed_noise_feats, renderer, render_mask, lpips_fn,
                                 ALL_FRAMES)
    held_mask  = [f in set(HELD_OUT) for f in ALL_FRAMES]
    train_mask = [f in set(TRAIN)    for f in ALL_FRAMES]

    held_psnrs  = [r['psnr']  for r, h in zip(final_res['per_frame'], held_mask)  if h]
    train_psnrs = [r['psnr']  for r, t in zip(final_res['per_frame'], train_mask) if t]
    held_ssims  = [r['ssim']  for r, h in zip(final_res['per_frame'], held_mask)  if h]
    train_ssims = [r['ssim']  for r, t in zip(final_res['per_frame'], train_mask) if t]
    held_lpipss = [r['lpips'] for r, h in zip(final_res['per_frame'], held_mask)  if h]
    train_lpipss= [r['lpips'] for r, t in zip(final_res['per_frame'], train_mask) if t]

    final_summary = {
        'all_frames': {
            'n': len(ALL_FRAMES),
            'psnr_mean': final_res['psnr_mean'],  'psnr_std': final_res['psnr_std'],
            'ssim_mean': final_res['ssim_mean'],  'ssim_std': final_res['ssim_std'],
            'lpips_mean': final_res['lpips_mean'],'lpips_std': final_res['lpips_std'],
        },
        'held_out': {
            'n': len(held_psnrs),
            'psnr_mean': float(np.mean(held_psnrs)),  'psnr_std': float(np.std(held_psnrs)),
            'ssim_mean': float(np.mean(held_ssims)),  'ssim_std': float(np.std(held_ssims)),
            'lpips_mean': float(np.mean(held_lpipss)),'lpips_std': float(np.std(held_lpipss)),
        },
        'train': {
            'n': len(train_psnrs),
            'psnr_mean': float(np.mean(train_psnrs)),  'psnr_std': float(np.std(train_psnrs)),
            'ssim_mean': float(np.mean(train_ssims)),  'ssim_std': float(np.std(train_ssims)),
            'lpips_mean': float(np.mean(train_lpipss)),'lpips_std': float(np.std(train_lpipss)),
        },
        'per_frame': final_res['per_frame'],
    }
    json.dump(final_summary, open(_OUT_DIR / 'final_eval.json', 'w'), indent=2)
    print(f'  All 150 : PSNR={final_summary["all_frames"]["psnr_mean"]:.3f}±{final_summary["all_frames"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["all_frames"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["all_frames"]["lpips_mean"]:.4f}')
    print(f'  Held-out: PSNR={final_summary["held_out"]["psnr_mean"]:.3f}±{final_summary["held_out"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["held_out"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["held_out"]["lpips_mean"]:.4f}')
    print(f'  Train   : PSNR={final_summary["train"]["psnr_mean"]:.3f}±{final_summary["train"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["train"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["train"]["lpips_mean"]:.4f}')
    print(f'  → final_eval.json saved')

    print(f'\n[DONE] {EPOCHS} epochs complete.')
    print(f'  best held-out PSNR : {best_psnr:.3f} dB')
    print(f'  results            : {_OUT_DIR}')


if __name__ == '__main__':
    main()
