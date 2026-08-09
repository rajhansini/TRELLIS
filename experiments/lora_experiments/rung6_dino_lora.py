"""
Rung 6 — DINOv2 Encoder LoRA (falsification arm).

Question: is the quality gap caused by the CONDITIONING (how the input frame is
turned into tokens by DINOv2) rather than generation (flow) or decode?

Config: freeze the flow model AND decoder. LoRA on the DINOv2 image encoder's
linear layers. Same loss, split, noise, metrics as every other rung.

Run only if Rungs 1-5 leave an unexplained gap (see spec Part B Rung 6).

DINOv2 architecture (dinov2_vitl14_reg, torch.hub):
  embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, mlp_hidden=4096
  Stored as model.blocks: flat nn.ModuleList (chunked_blocks=False by default)

Per-block adaptable linear layers:
  Layer          in    out    module path
  attn.qkv      1024  3072   blocks[i].attn.qkv   (fused QKV, nn.Linear)
  attn.proj     1024  1024   blocks[i].attn.proj  (output proj, nn.Linear)
  mlp.fc1       1024  4096   blocks[i].mlp.fc1    (nn.Linear)
  mlp.fc2       4096  1024   blocks[i].mlp.fc2    (nn.Linear)

Per-block params (rank r):
  attn only (qkv+proj):  (1024+3072+1024*2)*r = (4096+2048)*r = 6144r
  all 4 layers:          (4096+2048+5120+5120)*r = 16384r

All 24 blocks, rank 4:
  --layers attn  → 24 * 6144 * 4 = 589,824 params (spec default)
  --layers all   → 24 * 16384 * 4 = 1,572,864 params

Gradient path (longest of any rung):
  DINOv2+LoRA -> cond_gl -> [24-step prefix no-grad] -> last step with grad
  -> ns.feats -> slat -> slat_leaf_feats -> decode(frozen) -> render -> loss

Injection scheme (same memory trick as run01):
  1. encode_frame_with_lora(dino) -> cond_gl   [requires_grad via LoRA params]
  2. 24 prefix steps:  no_grad  (cond_gl used but graph not tracked)
  3. last step:        grad ON  (cond_gl now in computation graph)
  4. decode + render (decode detaches from flow via slat_leaf)
  5. loss.backward()  -> slat_leaf.grad
  6. torch.autograd.backward(slat.feats, slat_leaf.grad)
     -> propagates through last step -> cond_gl -> DINOv2 LoRA A,B

CRITICAL: do NOT touch or bypass the x_prenorm extraction or the subsequent
F.layer_norm in encode_frame_with_lora. Adapters change token values only.

Output: lora_experiments/runs/rung6_dino_{layers}_{blocks}_r{rank}_s{seed}_{hash}/
  lora_ckpts/lora_e{N:03d}.pt
  diag_renders/e{N:03d}/strip_f{F:04d}.png
  loss_history.json
  final_eval.json
  config.json
  train.log

Spec cross-ref: PART B RUNG 6; A.4 LoRA primitive; A.6 training mechanics;
                A.7 loss; A.8 data split; A.9 gates; A.10 logging.
"""

import sys, os, argparse as _ap
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

# ── arg parse (before output dir) ─────────────────────────────────────────────
_par = _ap.ArgumentParser()
_par.add_argument('--rank',   type=int, default=4)
_par.add_argument('--layers', choices=['attn', 'all'], default='attn',
                  help='attn=qkv+proj (spec default); all=+fc1+fc2')
_par.add_argument('--blocks', choices=['all', 'early', 'late'], default='all',
                  help='DINOv2 block range: all=0-23, early=0-11, late=12-23')
_par.add_argument('--epochs', type=int, default=20)
_par.add_argument('--seed',   type=int, default=6)
_par.add_argument('--lr',     type=float, default=1e-4,
                  help='learning rate; try 1e-3 if B.grad stays near zero')
_par.add_argument('--smoke',  action='store_true', help='10-frame dev run')
args = _par.parse_args()

import json, hashlib as _hlib
_CFG = dict(
    run='rung6',
    rank=args.rank,
    layers=args.layers,
    blocks=args.blocks,
    epochs=args.epochs,
    seed=args.seed,
    lr=args.lr,
    n_dino_blocks=24,
    dino_dim=1024,
    lora_alpha=args.rank,
)
_RUN_ID = _hlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_TAG     = f'rung6_dino_{args.layers}_{args.blocks}_r{args.rank}_s{args.seed}_{_RUN_ID}'
_OUT_DIR = _HERE / 'runs' / _TAG
_CKPT    = _OUT_DIR / 'lora_ckpts'
_DIAG    = _OUT_DIR / 'diag_renders'

for _d in (_OUT_DIR, _CKPT, _DIAG):
    _d.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(_OUT_DIR / 'train.log')
sys.stderr = sys.stdout


# ── nvdiffrast arch guard ────────────────────────────────────────────────────
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
        print(f'[NVDIFF] rebuilding: {e}', flush=True)
    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
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
LOSS_SCALE0 = 4096.0   # mandatory: longest gradient path of all rungs
W_LPIPS     = 0.1
DIAG_EVERY  = 5

HELD_OUT = list(range(5, 151, 10))    # 15 frames, NEVER trained on
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]  # 135 frames

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]

# ── Block range and layer set from CLI ───────────────────────────────────────
_DINO_BLOCKS = 24
_BLOCK_RANGES = {
    'all':   list(range(_DINO_BLOCKS)),
    'early': list(range(12)),
    'late':  list(range(12, _DINO_BLOCKS)),
}
ACTIVE_BLOCKS = _BLOCK_RANGES[args.blocks]
ACTIVE_LAYERS = {
    'attn': ('qkv', 'proj'),
    'all':  ('qkv', 'proj', 'fc1', 'fc2'),
}[args.layers]

# ── Param counts ─────────────────────────────────────────────────────────────
_DINO_DIM = 1024
_MLP_H    = _DINO_DIM * 4  # 4096
_PER_BLOCK = {
    'qkv':  (_DINO_DIM + _DINO_DIM * 3) * args.rank,    # 4096r
    'proj': (_DINO_DIM * 2) * args.rank,                  # 2048r
    'fc1':  (_DINO_DIM + _MLP_H) * args.rank,             # 5120r
    'fc2':  (_MLP_H + _DINO_DIM) * args.rank,             # 5120r
}
EXPECTED_PARAMS = len(ACTIVE_BLOCKS) * sum(_PER_BLOCK[l] for l in ACTIVE_LAYERS)

# ── LoRA primitive (A.4) ─────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """A.4 LoRA primitive: y = (α/r) B A x, fp32 compute."""
    def __init__(self, in_dim, out_dim, rank, lora_alpha=None):
        super().__init__()
        lora_alpha = rank if lora_alpha is None else lora_alpha
        self.scaling = lora_alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        d = x.dtype
        return (((x.float() @ self.A.float().T) @ self.B.float().T) * self.scaling).to(d)


class DinoLoRABlock(nn.Module):
    """LoRA adapters for one DINOv2 Block."""
    def __init__(self, dim=1024, rank=4, active_layers=('qkv', 'proj')):
        super().__init__()
        mlp_h = dim * 4
        self.lora_qkv  = LoRALayer(dim, dim * 3, rank) if 'qkv'  in active_layers else None
        self.lora_proj = LoRALayer(dim, dim,     rank) if 'proj' in active_layers else None
        self.lora_fc1  = LoRALayer(dim, mlp_h,   rank) if 'fc1'  in active_layers else None
        self.lora_fc2  = LoRALayer(mlp_h, dim,   rank) if 'fc2'  in active_layers else None

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── DINOv2 LoRA context manager ───────────────────────────────────────────────

def _make_plain_lora(orig_fwd, lora):
    """Wrap a plain-tensor nn.Linear forward with LoRA delta."""
    def _fwd(x):
        return orig_fwd(x) + lora(x)
    return _fwd


@contextmanager
def dino_lora_ctx(dino_model, lora_registry):
    """
    Patch active DINOv2 blocks with LoRA forwards.

    DINOv2 blocks[i] (Block / NestedTensorBlock):
      block.attn.qkv  : nn.Linear(1024, 3072) — fused QKV
      block.attn.proj : nn.Linear(1024, 1024) — output projection
      block.mlp.fc1   : nn.Linear(1024, 4096)
      block.mlp.fc2   : nn.Linear(4096, 1024)

    All receive plain tensors. Context manager replaces .forward on the INSTANCE
    to keep the class definition intact (backward compatible with eval modes).

    CRITICAL: does NOT touch the x_prenorm extraction point or the F.layer_norm
    applied to it by TRELLIS. Adapters only change intermediate DINOv2 activations.
    """
    saved = {}

    for b_idx in ACTIVE_BLOCKS:
        blk = dino_model.blocks[b_idx]
        lb  = lora_registry[b_idx]
        saved[b_idx] = {}

        if lb.lora_qkv is not None:
            saved[b_idx]['qkv']  = blk.attn.qkv.forward
            blk.attn.qkv.forward = _make_plain_lora(blk.attn.qkv.forward, lb.lora_qkv)

        if lb.lora_proj is not None:
            saved[b_idx]['proj'] = blk.attn.proj.forward
            blk.attn.proj.forward = _make_plain_lora(blk.attn.proj.forward, lb.lora_proj)

        if lb.lora_fc1 is not None:
            saved[b_idx]['fc1'] = blk.mlp.fc1.forward
            blk.mlp.fc1.forward = _make_plain_lora(blk.mlp.fc1.forward, lb.lora_fc1)

        if lb.lora_fc2 is not None:
            saved[b_idx]['fc2'] = blk.mlp.fc2.forward
            blk.mlp.fc2.forward = _make_plain_lora(blk.mlp.fc2.forward, lb.lora_fc2)

    try:
        yield
    finally:
        for b_idx, patches in saved.items():
            blk = dino_model.blocks[b_idx]
            if 'qkv'  in patches: blk.attn.qkv.forward   = patches['qkv']
            if 'proj' in patches: blk.attn.proj.forward  = patches['proj']
            if 'fc1'  in patches: blk.mlp.fc1.forward    = patches['fc1']
            if 'fc2'  in patches: blk.mlp.fc2.forward    = patches['fc2']


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt(frame_idx: int):
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt  = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    return gt, (gt < 0.99).any(dim=0)


def load_render_mask() -> torch.Tensor:
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


def _prep_image(frame_idx: int) -> torch.Tensor:
    """Load frame, resize 518x518, apply DINO normalize -> (1,3,518,518) on DEVICE."""
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((518, 518), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)


def encode_with_lora(dino_model, lora_registry, img_tensor, no_grad: bool = False):
    """
    DINOv2+LoRA encode. img_tensor: (1,3,518,518) on DEVICE.

    With no_grad=False (training): returns cond_gl (1374,1024) WITH gradient
    w.r.t. LoRA params, so torch.autograd.backward later propagates correctly.

    With no_grad=True (eval): same forward, no gradient bookkeeping.

    CRITICAL: x_prenorm extraction and F.layer_norm are NOT patched — only
    block.attn and block.mlp layers are modified by dino_lora_ctx.
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx, dino_lora_ctx(dino_model, lora_registry):
        feats = dino_model(img_tensor, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)   # (1374, 1024)


def encode_frozen(dino_model, img_tensor) -> torch.Tensor:
    """Frozen DINOv2 (no LoRA, no grad). For GATE 1 baseline comparison."""
    with torch.no_grad():
        feats = dino_model(img_tensor, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl):
    """All 25 steps no-grad. cond_gl may have requires_grad but graph not tracked."""
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(ns, t_ten, cond_gl)
            ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


# ── Norms ─────────────────────────────────────────────────────────────────────

def dino_block_norms(lora_registry, dino_model):
    """Per-block ||B@A|| / ||W|| for each active DINOv2 layer."""
    A_norms, B_norms, BA_norms, eff_ratios = [], [], [], []
    per_block = {}
    for b_idx in ACTIVE_BLOCKS:
        lb  = lora_registry[b_idx]
        blk = dino_model.blocks[b_idx]
        ratios_blk = {}
        pairs = []
        if lb.lora_qkv  is not None:
            pairs.append(('qkv',  lb.lora_qkv,  blk.attn.qkv.weight.float().norm().item()))
        if lb.lora_proj is not None:
            pairs.append(('proj', lb.lora_proj, blk.attn.proj.weight.float().norm().item()))
        if lb.lora_fc1  is not None:
            pairs.append(('fc1',  lb.lora_fc1,  blk.mlp.fc1.weight.float().norm().item()))
        if lb.lora_fc2  is not None:
            pairs.append(('fc2',  lb.lora_fc2,  blk.mlp.fc2.weight.float().norm().item()))
        for name, layer, wnorm in pairs:
            A = layer.A.float(); B = layer.B.float()
            ba = (B @ A).norm().item()
            A_norms.append(A.norm().item()); B_norms.append(B.norm().item())
            BA_norms.append(ba)
            ratio = ba / max(wnorm, 1e-8)
            eff_ratios.append(ratio); ratios_blk[name] = ratio
        per_block[b_idx] = ratios_blk
    return A_norms, B_norms, BA_norms, eff_ratios, per_block


def b_grad_norms(lora_registry):
    norms = []
    for b_idx in ACTIVE_BLOCKS:
        lb = lora_registry[b_idx]
        for layer in [lb.lora_qkv, lb.lora_proj, lb.lora_fc1, lb.lora_fc2]:
            if layer is not None and layer.B.grad is not None:
                norms.append(layer.B.grad.norm().item())
    return norms


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_frames(pipeline, flow_model, dino_model, lora_registry,
                    img_cache, coords, fixed_noise_feats,
                    renderer, render_mask, lpips_fn, frame_list):
    """
    Evaluate on frame_list with DINOv2+LoRA (no_grad).
    img_cache: dict frame_idx -> preprocessed (1,3,518,518) tensor on DEVICE.
    """
    per_frame = []
    for fi in frame_list:
        cond = encode_with_lora(dino_model, lora_registry, img_cache[fi], no_grad=True)
        cond_gl = cond.unsqueeze(0)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render   = color.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)
        psnr = masked_psnr(render, gt, render_mask)
        ssim = compute_ssim(render, gt)
        r_lp = render.unsqueeze(0) * 2 - 1
        g_lp = gt.unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(r_lp, g_lp).item()
        per_frame.append({'frame': fi, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
        del render, gt, gt_mask, slat, color
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


def evaluate_held_out(pipeline, flow_model, dino_model, lora_registry,
                      img_cache, coords, fixed_noise_feats,
                      renderer, render_mask, lpips_fn):
    res = evaluate_frames(pipeline, flow_model, dino_model, lora_registry,
                          img_cache, coords, fixed_noise_feats,
                          renderer, render_mask, lpips_fn, HELD_OUT)
    return res['psnr_mean'], res['psnr_std'], res['lpips_mean'], res['ssim_mean'], res


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


def save_diagnostics(pipeline, flow_model, dino_model, lora_registry,
                     img_cache, coords, fixed_noise_feats, renderer, epoch, raw_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    for fi in DIAG_FRAMES:
        cond = encode_with_lora(dino_model, lora_registry, img_cache[fi], no_grad=True)
        cond_gl = cond.unsqueeze(0)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render   = color.detach().clamp(0, 1)
        if fi not in raw_cache:
            raw_cache[fi] = render.clone()
        gt, _ = load_gt(fi)
        strip = make_strip([
            (raw_cache[fi], f'raw f{fi}'),
            (render,        f'lora e{epoch} f{fi}'),
            (gt,            f'GT f{fi}'),
        ])
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        del render, gt, slat, color
    gc.collect(); torch.cuda.empty_cache()


def save_loss_curve(history):
    if not history: return
    epochs = [r['epoch'] for r in history]
    psnrs  = [r['held_psnr'] for r in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, psnrs, 'o-', color='#61afef', lw=2, ms=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Held-out PSNR (dB)')
    ax.set_title(f'Rung 6 DINOv2-LoRA ({_TAG})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'psnr_curve.png', dpi=120, bbox_inches='tight')
    plt.close()


# ── Gates ─────────────────────────────────────────────────────────────────────

def gate0(lora_registry, dino_model):
    """GATE 0: param count, B==0, DINOv2 base params not trainable."""
    actual = sum(
        sum(p.numel() for p in lora_registry[b].parameters())
        for b in ACTIVE_BLOCKS
    )
    assert actual == EXPECTED_PARAMS, \
        f'GATE 0 FAIL: expected {EXPECTED_PARAMS}, got {actual}'

    for b in ACTIVE_BLOCKS:
        lb = lora_registry[b]
        for layer in [lb.lora_qkv, lb.lora_proj, lb.lora_fc1, lb.lora_fc2]:
            if layer is not None:
                assert (layer.B == 0).all(), 'GATE 0 FAIL: B not zero at init'

    for name, p in dino_model.named_parameters():
        if p.requires_grad:
            raise AssertionError(f'GATE 0 FAIL: DINOv2 param {name} has requires_grad=True')

    print(f'[GATE 0] PASSED: {actual:,} params, B=0 at init, DINOv2 frozen.')


def gate1(pipeline, flow_model, dino_model, lora_registry, img_cache,
          coords, fixed_noise_feats, renderer, render_mask):
    """GATE 1: B=0 → render == frozen DINOv2 TRELLIS within fp16 tol."""
    fi     = HELD_OUT[0]

    # Frozen DINOv2 (no LoRA)
    cond_frozen = encode_frozen(dino_model, img_cache[fi])
    slat_raw    = full_denoise_nograd(flow_model, fixed_noise_feats, coords,
                                      cond_frozen.unsqueeze(0))
    raw_color, _ = decode_and_render(pipeline, slat_raw, renderer, diag=False, device=DEVICE)
    raw_render   = raw_color.detach().clamp(0, 1)

    # DINOv2+LoRA (B=0, so delta==0)
    cond_lora = encode_with_lora(dino_model, lora_registry, img_cache[fi], no_grad=True)
    slat_lora = full_denoise_nograd(flow_model, fixed_noise_feats, coords,
                                    cond_lora.unsqueeze(0))
    lora_color, _ = decode_and_render(pipeline, slat_lora, renderer, diag=False, device=DEVICE)
    lora_render   = lora_color.detach().clamp(0, 1)

    # Token-level check too
    tok_diff  = (cond_frozen - cond_lora).abs().max().item()
    pix_diff  = (raw_render - lora_render).abs().max().item()
    assert pix_diff < 1e-3, f'GATE 1 FAIL: max|raw-lora| = {pix_diff:.3e} > 1e-3'
    print(f'[GATE 1] PASSED: token max_diff={tok_diff:.2e}  pixel max_diff={pix_diff:.2e}')
    del raw_render, lora_render
    gc.collect(); torch.cuda.empty_cache()


def gate2_check(lora_registry):
    """GATE 2: check DINOv2 LoRA B matrices have nonzero finite gradients."""
    issues = []
    for b_idx in ACTIVE_BLOCKS:
        lb = lora_registry[b_idx]
        for name, layer in [('qkv', lb.lora_qkv), ('proj', lb.lora_proj),
                             ('fc1', lb.lora_fc1), ('fc2', lb.lora_fc2)]:
            if layer is None: continue
            g = layer.B.grad
            if g is None:
                issues.append(f'b{b_idx}.{name}: B.grad is None')
            elif not torch.isfinite(g).all():
                issues.append(f'b{b_idx}.{name}: B.grad has NaN/Inf')
            elif g.abs().max().item() == 0.0:
                issues.append(f'b{b_idx}.{name}: B.grad is all zeros')
    if issues:
        print('[GATE 2] FAIL — per-block gradient diagnostic:')
        for s in issues[:10]: print(f'  {s}')
        if len(issues) > 10: print(f'  ... ({len(issues)} total)')
        print('  NOTE: for Rung 6, B.grad arrives via the longest path')
        print('        (encode->25-step flow->decode->render). If all None,')
        print('        check that encode_with_lora runs WITHOUT no_grad during training.')
        raise RuntimeError('GATE 2 FAILED — gradients not flowing to DINOv2 LoRA')
    print('[GATE 2] PASSED: all B matrices have finite nonzero gradients.')


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print(f'Rung 6 — DINOv2 Encoder LoRA')
    print(f'  tag            : {_TAG}')
    print(f'  run_id         : {_RUN_ID}')
    print(f'  rank           : {args.rank}')
    print(f'  layers         : {ACTIVE_LAYERS}')
    print(f'  blocks         : {args.blocks} ({ACTIVE_BLOCKS})')
    print(f'  lr             : {args.lr}')
    print(f'  epochs         : {args.epochs}')
    print(f'  seed           : {args.seed}')
    print(f'  expected_params: {EXPECTED_PARAMS:,}')
    print(f'  output         : {_OUT_DIR}')
    print('=' * 70)
    print()
    print('WARNING: Rung 6 should only run if Rungs 1-5 leave an unexplained gap.')
    print('         Gradient path is the longest of all rungs (DINOv2 -> flow -> decode).')
    print('         If B.grad stays near zero, try --lr 1e-3.')
    print()

    json.dump(
        {**_CFG, 'run_id': _RUN_ID, 'expected_params': EXPECTED_PARAMS,
         'active_blocks': ACTIVE_BLOCKS, 'active_layers': list(ACTIVE_LAYERS)},
        open(_OUT_DIR / 'config.json', 'w'), indent=2
    )

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print(f'[SETUP] Loading pipeline from {PRETRAINED}...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)

    # Freeze ALL pipeline parameters — LoRA registry is separate
    for p in pipeline.parameters():
        p.requires_grad_(False)
    flow_model.eval(); dec_model.eval(); dino_model.eval()

    # Offload unused models
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except: pass
    torch.cuda.empty_cache()

    # ── Sparse structure ──────────────────────────────────────────────────────
    print(f'[STRUCT] Sampling structure (seed={STRUCT_SEED})...')
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}')
    del cond_struct
    gc.collect(); torch.cuda.empty_cache()

    # ── Fixed noise ───────────────────────────────────────────────────────────
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'[NOISE] Fixed noise seed={FIXED_SEED}  shape={tuple(fixed_noise_feats.shape)}')

    # ── Pre-process all frames (resize+normalize, store on CPU) ───────────────
    # Unlike other rungs, Rung 6 cannot pre-compute DINOv2 tokens (LoRA is active).
    # Instead we cache preprocessed image tensors and re-encode each epoch.
    print(f'\n[IMG CACHE] Pre-processing {N_FRAMES} frames (resize+normalize, CPU)...')
    img_cache = {}
    for i in range(1, N_FRAMES + 1):
        img_cache[i] = _prep_image(i).cpu()   # (1,3,518,518) float32 on CPU
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    cache_mb = sum(t.numel() * 4 for t in img_cache.values()) / 1e6
    print(f'  Cache size: {cache_mb:.0f} MB')

    # Bring each frame to DEVICE on-the-fly during training
    def get_img(frame_idx):
        return img_cache[frame_idx].to(DEVICE)

    # ── Renderer & mask ───────────────────────────────────────────────────────
    renderer    = make_renderer()
    render_mask = load_render_mask()
    n_mask      = render_mask.sum().item()
    print(f'[MASK] {n_mask} masked pixels ({100*n_mask/RENDER_RES**2:.1f}%)')

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    print('[LPIPS] Loading AlexNet...')
    import lpips as _lpips
    lpips_fn = _lpips.LPIPS(net='alex').to(DEVICE); lpips_fn.eval()

    # ── Build DINOv2 LoRA registry ────────────────────────────────────────────
    lora_registry = nn.ModuleDict({
        str(b): DinoLoRABlock(dim=_DINO_DIM, rank=args.rank, active_layers=ACTIVE_LAYERS)
        for b in ACTIVE_BLOCKS
    })
    lora_registry = lora_registry.to(DEVICE)
    type(lora_registry).__getitem__ = lambda self, k: self._modules[str(k)]

    lora_params = list(lora_registry.parameters())
    print(f'[LORA] Registry built:')
    print(f'  active blocks   : {ACTIVE_BLOCKS}')
    print(f'  active layers   : {list(ACTIVE_LAYERS)}')
    print(f'  trainable params: {sum(p.numel() for p in lora_params):,}')

    # ── GATE 0 (runs before checkpoint load so B=0 check is valid) ───────────
    gate0(lora_registry, dino_model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt    = torch.optim.Adam(lora_params, lr=args.lr, eps=1e-16, weight_decay=0.0)
    EPOCHS = 2 if args.smoke else args.epochs

    # ── Resume logic ──────────────────────────────────────────────────────────
    def _find_latest_ckpt():
        ckpts = sorted(_CKPT.glob('lora_e[0-9][0-9][0-9].pt'))
        return ckpts[-1] if ckpts else None

    start_epoch = 1
    best_psnr   = -999.0
    history     = []
    loss_scale  = LOSS_SCALE0

    latest_ckpt = _find_latest_ckpt()
    resumed     = latest_ckpt is not None
    if latest_ckpt:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        assert ckpt.get('run_id') == _RUN_ID, \
            f'Checkpoint run_id mismatch: {ckpt.get("run_id")} != {_RUN_ID}'
        lora_registry.load_state_dict(ckpt['lora_state'])
        opt.load_state_dict(ckpt['opt_state'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -999.0)
        loss_scale  = ckpt.get('loss_scale', LOSS_SCALE0)
        hist_path   = _OUT_DIR / 'loss_history.json'
        if hist_path.exists():
            history = json.load(open(hist_path))
        print(f'  resumed from epoch {ckpt["epoch"]}  '
              f'best_psnr={best_psnr:.3f}  loss_scale={loss_scale:.0f}')
    else:
        print('\n[RESUME] No checkpoint found — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    # ── GATE 1 (fresh runs only; skip on resume since B ≠ 0 after load) ───────
    if resumed:
        print('\n[GATE 1] SKIPPED — resuming from checkpoint (B ≠ 0 by design)\n')
    else:
        gate1(pipeline, flow_model, dino_model, lora_registry,
              {fi: get_img(fi) for fi in [HELD_OUT[0]]},
              coords, fixed_noise_feats, renderer, render_mask)

    raw_cache  = {}
    gate2_done = False
    frame_list = TRAIN if not args.smoke else TRAIN[:10]

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep    = time.time()
        random.shuffle(frame_list)
        n_ok    = 0
        tot_mse = tot_lp = tot_tot = 0.0

        for frame_i in frame_list:
            opt.zero_grad(set_to_none=True)
            img_t = get_img(frame_i)   # (1,3,518,518) on DEVICE

            # Step 1: encode with DINOv2+LoRA (grad enabled on LoRA params)
            cond = encode_with_lora(dino_model, lora_registry, img_t, no_grad=False)
            cond_gl = cond.unsqueeze(0)   # (1, 1374, 1024), requires_grad=True

            # Step 2: 24 prefix denoising steps — no grad on activations,
            # but cond_gl IS used as conditioning (graph not tracked for prefix steps)
            x_in = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
            with torch.no_grad():
                for t, t_prev in T_PAIRS[:-1]:
                    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                    v     = flow_model(x_in, t_ten, cond_gl)
                    x_in  = x_in.replace(x_in.feats - (t - t_prev) * v.feats)

            # Step 3: last denoising step WITH grad — cond_gl enters the graph here
            t_last, t_prev_last = T_PAIRS[-1]
            t_ten = torch.tensor([1000.0 * t_last], device=DEVICE, dtype=torch.float32)
            v     = flow_model(x_in, t_ten, cond_gl)   # cond_gl in grad graph now
            ns    = x_in.replace(x_in.feats - (t_last - t_prev_last) * v.feats)

            # Step 4: decode + render (slat_leaf detaches from flow)
            slat  = normalize_slat(ns)
            color, slat_leaf = decode_and_render(pipeline, slat, renderer,
                                                 diag=False, device=DEVICE)

            # Step 5: loss
            gt, gt_mask = load_gt(frame_i)
            mse_v, lp_v, loss_v = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)

            # loss.backward() → slat_leaf.grad (gradient at slat_leaf_feats)
            (loss_v * loss_scale).backward()

            # Step 6: flow backward injection — propagates slat_leaf.grad back
            # through normalize_slat and the last denoising step to cond_gl,
            # then to DINOv2 LoRA params (A, B)
            if slat_leaf.grad is not None:
                torch.autograd.backward(slat.feats, slat_leaf.grad)
            # Now lora_params have .grad populated

            # GATE 2: check after first backward
            if not gate2_done:
                gate2_check(lora_registry)
                gate2_done = True

            # Step 7: unscale LoRA grads (LOSS_SCALE applied in step 5)
            for p in lora_params:
                if p.grad is not None:
                    p.grad.div_(loss_scale)

            all_finite = all(
                torch.isfinite(p.grad).all()
                for p in lora_params if p.grad is not None
            )
            if not all_finite:
                opt.zero_grad(set_to_none=True)
                loss_scale = max(loss_scale / 2, 1.0)
                print(f'  [SCALE DOWN] loss_scale -> {loss_scale:.0f}')
                continue

            opt.step()
            n_ok    += 1
            tot_mse += mse_v.item()
            tot_lp  += lp_v.item()
            tot_tot += loss_v.item()

            del color, slat_leaf, gt, gt_mask, mse_v, lp_v, loss_v
            del slat, ns, v, cond, cond_gl, x_in, img_t
            gc.collect(); torch.cuda.empty_cache()

        avg_mse = tot_mse / max(n_ok, 1)
        avg_lp  = tot_lp  / max(n_ok, 1)
        avg_tot = tot_tot / max(n_ok, 1)

        # Norms
        A_norms, B_norms, BA_norms, eff_ratios, per_block = dino_block_norms(
            lora_registry, dino_model)

        # Held-out eval (re-encodes with LoRA but no_grad)
        held_psnr, held_psnr_std, held_lpips, held_ssim, held_res = evaluate_held_out(
            pipeline, flow_model, dino_model, lora_registry,
            {fi: get_img(fi) for fi in HELD_OUT},
            coords, fixed_noise_feats, renderer, render_mask, lpips_fn)

        is_best = held_psnr > best_psnr
        if is_best:
            best_psnr = held_psnr

        _ckpt_data = {
            'run_id': _RUN_ID, 'run': 'rung6', 'epoch': epoch,
            'lora_state': lora_registry.state_dict(),
            'opt_state': opt.state_dict(),
            'best_psnr': best_psnr, 'held_psnr': held_psnr,
            'loss_scale': loss_scale,
        }
        torch.save(_ckpt_data, _CKPT / f'lora_e{epoch:03d}.pt')
        if is_best:
            torch.save(_ckpt_data, _CKPT / 'lora_best.pt')
            print(f'  [CKPT] new best → lora_best.pt')

        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        elapsed = time.time() - t_ep
        b_gn    = b_grad_norms(lora_registry)

        print(f'\n[EPOCH {epoch:02d}]  loss={avg_tot:.5f}  '
              f'(mse={avg_mse:.5f}  lpips×{W_LPIPS}={W_LPIPS*avg_lp:.5f})')
        print(f'  ||A|| mean={np.mean(A_norms):.4f}  '
              f'min={np.min(A_norms):.4f}  max={np.max(A_norms):.4f}')
        print(f'  ||B|| mean={np.mean(B_norms):.4f}  '
              f'min={np.min(B_norms):.4f}  max={np.max(B_norms):.4f}')
        print(f'  ||B@A||/||W|| mean={np.mean(eff_ratios):.4f}  '
              f'min={np.min(eff_ratios):.4f}  max={np.max(eff_ratios):.4f}')
        if b_gn:
            print(f'  B.grad norms: min={min(b_gn):.3e}  max={max(b_gn):.3e}')
        else:
            print(f'  B.grad norms: [none — gradient may not be reaching LoRA params]')
        print(f'  held PSNR={held_psnr:.3f}±{held_psnr_std:.3f} dB  '
              f'SSIM={held_ssim:.4f}  LPIPS={held_lpips:.4f}  '
              f'loss_scale={loss_scale:.0f}  GPU={gpu_mem_gb:.1f}GB  '
              f'{"BEST ✓" if is_best else ""}  time={elapsed:.1f}s\n')

        row = {
            'epoch': epoch, 'n_steps': n_ok, 'loss_total': avg_tot,
            'loss_mse': avg_mse, 'loss_lpips': avg_lp,
            'held_psnr': held_psnr, 'held_psnr_std': held_psnr_std,
            'held_ssim': held_ssim, 'held_ssim_std': held_res['ssim_std'],
            'held_lpips': held_lpips, 'held_lpips_std': held_res['lpips_std'],
            'held_per_frame': held_res['per_frame'],
            'A_norm_mean': float(np.mean(A_norms)), 'A_norm_min': float(np.min(A_norms)),
            'A_norm_max': float(np.max(A_norms)),
            'B_norm_mean': float(np.mean(B_norms)), 'B_norm_min': float(np.min(B_norms)),
            'B_norm_max': float(np.max(B_norms)),
            'BA_norm_mean': float(np.mean(BA_norms)),
            'eff_ratio_mean': float(np.mean(eff_ratios)),
            'eff_ratio_min':  float(np.min(eff_ratios)),
            'eff_ratio_max':  float(np.max(eff_ratios)),
            'per_block_ratios': {str(k): v for k, v in per_block.items()},
            'B_grad_min': float(min(b_gn)) if b_gn else None,
            'B_grad_max': float(max(b_gn)) if b_gn else None,
            'loss_scale': loss_scale,
            'peak_gpu_gb': round(gpu_mem_gb, 3),
            'wall_secs': elapsed,
        }
        history.append(row)
        json.dump(history, open(_OUT_DIR / 'loss_history.json', 'w'), indent=2)

        if epoch % DIAG_EVERY == 0 or epoch == EPOCHS:
            save_diagnostics(pipeline, flow_model, dino_model, lora_registry,
                             {fi: get_img(fi) for fi in DIAG_FRAMES},
                             coords, fixed_noise_feats, renderer, epoch, raw_cache)
            save_loss_curve(history)
            gc.collect(); torch.cuda.empty_cache()

    # ── Per-block BA_ratio figure ─────────────────────────────────────────────
    if history and 'per_block_ratios' in history[-1]:
        last   = history[-1]['per_block_ratios']
        blocks = sorted(int(k) for k in last.keys())
        fig, axes = plt.subplots(1, len(ACTIVE_LAYERS),
                                 figsize=(5 * len(ACTIVE_LAYERS), 4), squeeze=False)
        colors = {'qkv': '#98c379', 'proj': '#61afef', 'fc1': '#e06c75', 'fc2': '#d19a66'}
        for col, lname in enumerate(ACTIVE_LAYERS):
            ax = axes[0][col]
            vals = [last[str(b)].get(lname, 0.0) for b in blocks]
            ax.bar(blocks, vals, color=colors.get(lname, '#abb2bf'))
            ax.axhline(0.1,  color='#e06c75', lw=1, ls='--', label='ceiling (0.1)')
            ax.axhline(0.01, color='#d19a66', lw=1, ls=':',  label='floor (0.01)')
            ax.set_xlabel('DINOv2 block index'); ax.set_ylabel('||B@A|| / ||W||')
            ax.set_title(f'dino.{lname}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f'Rung 6 — Per-block DINOv2 correction magnitude (epoch {EPOCHS})')
        fig.tight_layout()
        fig.savefig(_DIAG / 'per_block_BA_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  per-block figure : {_DIAG}/per_block_BA_ratio.png')

    # ── Final 150-frame eval ─────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('[FINAL EVAL] All 150 frames — post-training quality audit')
    print('=' * 70)
    ALL_FRAMES = list(range(1, N_FRAMES + 1))
    final_res  = evaluate_frames(pipeline, flow_model, dino_model, lora_registry,
                                 {fi: get_img(fi) for fi in ALL_FRAMES},
                                 coords, fixed_noise_feats, renderer,
                                 render_mask, lpips_fn, ALL_FRAMES)

    held_mask  = [f in set(HELD_OUT) for f in ALL_FRAMES]
    train_mask = [f in set(TRAIN)    for f in ALL_FRAMES]

    def _split(key):
        held  = [r[key] for r, h in zip(final_res['per_frame'], held_mask)  if h]
        train = [r[key] for r, t in zip(final_res['per_frame'], train_mask) if t]
        return held, train

    hp, tp = _split('psnr'); hs, ts = _split('ssim'); hl, tl = _split('lpips')
    final_summary = {
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
    json.dump(final_summary, open(_OUT_DIR / 'final_eval.json', 'w'), indent=2)
    print(f'  All 150: PSNR={final_summary["all"]["psnr_mean"]:.3f}±'
          f'{final_summary["all"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["all"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["all"]["lpips_mean"]:.4f}')
    print(f'  Held:    PSNR={final_summary["held"]["psnr_mean"]:.3f}±'
          f'{final_summary["held"]["psnr_std"]:.3f} dB'
          f'  SSIM={final_summary["held"]["ssim_mean"]:.4f}')
    print(f'  Train:   PSNR={final_summary["train"]["psnr_mean"]:.3f}±'
          f'{final_summary["train"]["psnr_std"]:.3f} dB')

    print(f'\n[DONE] {EPOCHS} epochs complete.')
    print(f'  best held-out PSNR : {best_psnr:.3f} dB')
    print(f'  config             : {_OUT_DIR}/config.json  (MD5: {_RUN_ID})')
    print(f'  results            : {_OUT_DIR}')


if __name__ == '__main__':
    main()
