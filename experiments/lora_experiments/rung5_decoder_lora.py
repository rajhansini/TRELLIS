"""
Rung 5 — Decoder LoRA (parallel diagnostic arm).

Question: does the appearance gap live in the LATENT (flow model) or in DECODE
(latent -> mesh -> render)?

Config: freeze the flow model entirely. LoRA on the SLAT decoder's linear layers.
Same loss, split, noise, metrics as every other rung.

Decoder architecture (SLatMeshDecoder):
  model_channels=768, latent_channels=8, num_blocks=12, num_heads=12, mlp_ratio=4
  use_fp16=True (decoder weights fp16; LoRA computed fp32 per spec)

Per-block adaptable linear layers:
  Layer          in    out    shape                rank-r params
  attn.to_qkv   768  2304   nn.Linear             (768+2304)*r = 3072r
  attn.to_out   768   768   nn.Linear             (768+768)*r  = 1536r
  mlp.fc1       768  3072   SparseLinear          (768+3072)*r = 3840r
  mlp.fc2      3072   768   SparseLinear          (3072+768)*r = 3840r

Per-block totals (rank 4):
  attn only (to_qkv+to_out):  4*4608  = 18,432
  all 4 layers:                4*12288 = 49,152

All 12 blocks:
  --layers attn  → 12 * 18,432 = 221,184 params
  --layers all   → 12 * 49,152 = 589,824 params (default)

Gradient path: loss -> render -> decode (decoder LoRA) -> slat_leaf_feats
  No flow backward injection needed — decoder LoRA params get gradients
  directly from loss.backward() since they sit between slat_leaf_feats and
  the rendered color.

Output: lora_experiments/runs/rung5_dec_{layers}_{blocks}_r{rank}_s{seed}_{hash}/
  lora_ckpts/lora_e{N:03d}.pt
  diag_renders/e{N:03d}/strip_f{F:04d}.png
  loss_history.json
  final_eval.json
  config.json
  train.log

Spec cross-ref: PART B RUNG 5; A.4 LoRA primitive; A.6 training mechanics;
                A.7 loss; A.8 data split; A.9 gates; A.10 logging.
"""

import sys, os, argparse as _ap, hashlib
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent   # lora_experiments/
_ROOT = _HERE.parent.parent               # TRELLIS/
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

# ── arg parse (early, before output dir) ─────────────────────────────────────
_par = _ap.ArgumentParser()
_par.add_argument('--rank',   type=int, default=4)
_par.add_argument('--layers', choices=['attn', 'all'], default='all',
                  help='attn=to_qkv+to_out; all=+fc1+fc2')
_par.add_argument('--blocks', choices=['all', 'early', 'mid', 'late'], default='all',
                  help='decoder block range: all=0-11, early=0-5, mid=3-8, late=6-11')
_par.add_argument('--epochs', type=int, default=20)
_par.add_argument('--seed',   type=int, default=6)
_par.add_argument('--smoke',  action='store_true', help='10-frame dev run')
args = _par.parse_args()

import json, hashlib as _hlib
_CFG = dict(
    run='rung5',
    rank=args.rank,
    layers=args.layers,
    blocks=args.blocks,
    epochs=args.epochs,
    seed=args.seed,
    n_dec_blocks=12,
    dec_dim=768,
    lora_alpha=args.rank,
)
_RUN_ID = _hlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_TAG    = f'rung5_dec_{args.layers}_{args.blocks}_r{args.rank}_s{args.seed}_{_RUN_ID}'
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
LR          = 1e-4
LOSS_SCALE0 = 4096.0       # safety — decoder grads are larger than flow grads
W_LPIPS     = 0.1
DIAG_EVERY  = 5

HELD_OUT = list(range(5, 151, 10))     # 15 frames, never trained on
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]   # 135 frames

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]

# ── Block range from CLI ──────────────────────────────────────────────────────
_DEC_BLOCKS = 12
_BLOCK_RANGES = {
    'all':   list(range(_DEC_BLOCKS)),    # 0-11
    'early': list(range(6)),              # 0-5
    'mid':   list(range(3, 9)),           # 3-8
    'late':  list(range(6, _DEC_BLOCKS)), # 6-11
}
ACTIVE_BLOCKS = _BLOCK_RANGES[args.blocks]
ACTIVE_LAYERS = {
    'attn': ('qkv', 'out'),
    'all':  ('qkv', 'out', 'fc1', 'fc2'),
}[args.layers]

# ── Param counts (verified against config) ───────────────────────────────────
_DEC_DIM = 768
_MLP_H   = _DEC_DIM * 4    # 3072
_PER_BLOCK = {
    'qkv': (_DEC_DIM + _DEC_DIM * 3) * args.rank,   # 3072r
    'out': (_DEC_DIM * 2) * args.rank,               # 1536r
    'fc1': (_DEC_DIM + _MLP_H) * args.rank,          # 3840r
    'fc2': (_MLP_H + _DEC_DIM) * args.rank,          # 3840r
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


class DecoderLoRABlock(nn.Module):
    """LoRA adapters for one decoder SparseTransformerBlock."""
    def __init__(self, dim=768, rank=4, active_layers=('qkv', 'out', 'fc1', 'fc2')):
        super().__init__()
        mlp_h = dim * 4
        self.lora_qkv = LoRALayer(dim, dim * 3, rank) if 'qkv' in active_layers else None
        self.lora_out = LoRALayer(dim, dim,     rank) if 'out' in active_layers else None
        self.lora_fc1 = LoRALayer(dim, mlp_h,   rank) if 'fc1' in active_layers else None
        self.lora_fc2 = LoRALayer(mlp_h, dim,   rank) if 'fc2' in active_layers else None

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Decoder LoRA context manager ─────────────────────────────────────────────

@contextmanager
def decoder_lora_ctx(dec_model, lora_registry):
    """
    Patch all active decoder blocks with LoRA forwards.

    dec_model.blocks[i] is SparseTransformerBlock (SELF-attention only):
      block.attn.to_qkv : nn.Linear(768, 2304)  — receives plain tensor from _linear()
      block.attn.to_out : nn.Linear(768, 768)   — receives plain tensor
      block.mlp.mlp[0]  : SparseLinear(768, 3072) — receives SparseTensor
      block.mlp.mlp[2]  : SparseLinear(3072, 768) — receives SparseTensor

    For to_qkv / to_out: SparseMultiHeadAttention._linear() extracts .feats
    before calling the module, so patched forward receives a plain tensor.

    For SparseLinear (mlp): Sequential passes SparseTensor directly; patched
    forward wraps the output SparseTensor.
    """
    saved = {}  # block_idx -> {attr_path: original_forward}

    def _make_plain_lora(orig_fwd, lora):
        """Patch for plain-tensor linear layers (to_qkv, to_out)."""
        def _fwd(x):
            return orig_fwd(x) + lora(x)
        return _fwd

    def _make_sparse_lora(orig_fwd, lora):
        """Patch for SparseLinear (mlp.mlp[0], mlp.mlp[2])."""
        def _fwd(x):   # x is SparseTensor
            out = orig_fwd(x)   # SparseTensor
            return out.replace(out.feats + lora(x.feats))
        return _fwd

    for b_idx in ACTIVE_BLOCKS:
        blk = dec_model.blocks[b_idx]
        lb  = lora_registry[b_idx]
        saved[b_idx] = {}

        if lb.lora_qkv is not None:
            saved[b_idx]['qkv']  = blk.attn.to_qkv.forward
            blk.attn.to_qkv.forward = _make_plain_lora(blk.attn.to_qkv.forward, lb.lora_qkv)

        if lb.lora_out is not None:
            saved[b_idx]['out']  = blk.attn.to_out.forward
            blk.attn.to_out.forward = _make_plain_lora(blk.attn.to_out.forward, lb.lora_out)

        if lb.lora_fc1 is not None:
            saved[b_idx]['fc1']  = blk.mlp.mlp[0].forward
            blk.mlp.mlp[0].forward = _make_sparse_lora(blk.mlp.mlp[0].forward, lb.lora_fc1)

        if lb.lora_fc2 is not None:
            saved[b_idx]['fc2']  = blk.mlp.mlp[2].forward
            blk.mlp.mlp[2].forward = _make_sparse_lora(blk.mlp.mlp[2].forward, lb.lora_fc2)

    try:
        yield
    finally:
        for b_idx, patches in saved.items():
            blk = dec_model.blocks[b_idx]
            if 'qkv' in patches: blk.attn.to_qkv.forward = patches['qkv']
            if 'out' in patches: blk.attn.to_out.forward = patches['out']
            if 'fc1' in patches: blk.mlp.mlp[0].forward  = patches['fc1']
            if 'fc2' in patches: blk.mlp.mlp[2].forward  = patches['fc2']


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    """Frozen DINOv2 encode — no grad. Pre-computed once."""
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


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


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl):
    """All 25 steps no-grad. Returns normalize_slat(result)."""
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(ns, t_ten, cond_gl)
            ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


# ── Norms ─────────────────────────────────────────────────────────────────────

def dec_block_norms(lora_registry, dec_model):
    """Per-block ||B@A|| / ||W|| for each active layer."""
    A_norms, B_norms, BA_norms, eff_ratios = [], [], [], []
    per_block = {}
    for b_idx in ACTIVE_BLOCKS:
        lb  = lora_registry[b_idx]
        blk = dec_model.blocks[b_idx]
        ratios_blk = {}
        pairs = []
        if lb.lora_qkv is not None:
            pairs.append(('qkv', lb.lora_qkv, blk.attn.to_qkv.weight.float().norm().item()))
        if lb.lora_out is not None:
            pairs.append(('out', lb.lora_out, blk.attn.to_out.weight.float().norm().item()))
        if lb.lora_fc1 is not None:
            pairs.append(('fc1', lb.lora_fc1, blk.mlp.mlp[0].weight.float().norm().item()))
        if lb.lora_fc2 is not None:
            pairs.append(('fc2', lb.lora_fc2, blk.mlp.mlp[2].weight.float().norm().item()))
        for name, layer, wnorm in pairs:
            A = layer.A.float(); B = layer.B.float()
            ba = (B @ A).norm().item()
            A_norms.append(A.norm().item())
            B_norms.append(B.norm().item())
            BA_norms.append(ba)
            ratio = ba / max(wnorm, 1e-8)
            eff_ratios.append(ratio)
            ratios_blk[name] = ratio
        per_block[b_idx] = ratios_blk
    return A_norms, B_norms, BA_norms, eff_ratios, per_block


def b_grad_norms(lora_registry):
    norms = []
    for b_idx in ACTIVE_BLOCKS:
        lb = lora_registry[b_idx]
        for layer in [lb.lora_qkv, lb.lora_out, lb.lora_fc1, lb.lora_fc2]:
            if layer is not None and layer.B.grad is not None:
                norms.append(layer.B.grad.norm().item())
    return norms


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_frames(pipeline, flow_model, dec_model, lora_registry,
                    raw_tokens, coords, fixed_noise_feats,
                    renderer, render_mask, lpips_fn, frame_list):
    per_frame = []
    for fi in frame_list:
        cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
        with decoder_lora_ctx(dec_model, lora_registry):
            color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render = color.detach().clamp(0, 1)
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


def evaluate_held_out(pipeline, flow_model, dec_model, lora_registry,
                      raw_tokens, coords, fixed_noise_feats,
                      renderer, render_mask, lpips_fn):
    res = evaluate_frames(pipeline, flow_model, dec_model, lora_registry,
                          raw_tokens, coords, fixed_noise_feats,
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


def save_diagnostics(pipeline, flow_model, dec_model, lora_registry,
                     raw_tokens, coords, fixed_noise_feats, renderer, epoch, raw_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    for fi in DIAG_FRAMES:
        cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
        with decoder_lora_ctx(dec_model, lora_registry):
            color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        render = color.detach().clamp(0, 1)
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
    if not history:
        return
    epochs = [r['epoch'] for r in history]
    psnrs  = [r['held_psnr'] for r in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, psnrs, 'o-', color='#98c379', lw=2, ms=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Held-out PSNR (dB)')
    ax.set_title(f'Rung 5 Dec-LoRA ({_TAG})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'psnr_curve.png', dpi=120, bbox_inches='tight')
    plt.close()


# ── Gates ─────────────────────────────────────────────────────────────────────

def gate0(lora_registry, dec_model):
    """GATE 0: N_dec check, param count, B==0, no frozen param has requires_grad."""
    from trellis.modules.sparse.transformer.blocks import SparseTransformerBlock as _STB
    n_dec_actual = len([m for _, m in dec_model.named_modules() if isinstance(m, _STB)])
    assert n_dec_actual == _DEC_BLOCKS, \
        f'GATE 0 FAIL: expected {_DEC_BLOCKS} decoder blocks, found {n_dec_actual}. ' \
        f'Update _DEC_BLOCKS in the script.'
    print(f'[GATE 0] N_dec confirmed: {n_dec_actual} SparseTransformerBlocks')

    actual = sum(
        sum(p.numel() for p in lora_registry[b].parameters())
        for b in ACTIVE_BLOCKS
    )
    assert actual == EXPECTED_PARAMS, \
        f'GATE 0 FAIL: expected {EXPECTED_PARAMS}, got {actual}'

    for b in ACTIVE_BLOCKS:
        lb = lora_registry[b]
        for layer in [lb.lora_qkv, lb.lora_out, lb.lora_fc1, lb.lora_fc2]:
            if layer is not None:
                assert (layer.B == 0).all(), 'GATE 0 FAIL: B not zero at init'

    # Frozen decoder params: must not have requires_grad
    # (LoRA params live in lora_registry, NOT inside dec_model)
    for name, p in dec_model.named_parameters():
        if p.requires_grad:
            raise AssertionError(f'GATE 0 FAIL: decoder param {name} has requires_grad=True')

    print('[GATE 0] PASSED: param count OK, B=0 at init, decoder frozen.')


def gate1(pipeline, flow_model, dec_model, lora_registry, raw_tokens,
          coords, fixed_noise_feats, renderer, render_mask):
    """GATE 1-plain + GATE 1-sparse: identity at init checks."""
    fi      = HELD_OUT[0]
    cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)

    # ── GATE 1-sparse: coordinate preservation (spec: THE critical one) ────────
    # Hook dec_model.out_layer (last SparseLinear before mesh extraction) to
    # capture the output SparseTensor coordinates. LoRA only modifies features;
    # coords must be bit-identical with and without B=0 LoRA. Zero tolerance.
    slat_coord = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
    slat_input = sp.SparseTensor(
        feats=slat_coord.feats.detach().clone(), coords=slat_coord.coords)
    _captured = {}

    def _coord_hook(key):
        def _h(module, inp, out):
            if isinstance(out, sp.SparseTensor):
                _captured[key] = out.coords.clone()
        return _h

    with torch.no_grad():
        h1 = dec_model.out_layer.register_forward_hook(_coord_hook('frozen'))
        dec_model(slat_input)
        h1.remove()

    with decoder_lora_ctx(dec_model, lora_registry):
        with torch.no_grad():
            h2 = dec_model.out_layer.register_forward_hook(_coord_hook('lora'))
            dec_model(slat_input)
            h2.remove()

    assert 'frozen' in _captured and 'lora' in _captured, \
        'GATE 1-sparse: forward hook did not fire — check out_layer attribute'
    assert torch.equal(_captured['frozen'], _captured['lora']), (
        f'GATE 1-sparse FAIL: coordinate mismatch! '
        f'frozen={_captured["frozen"].shape} lora={_captured["lora"].shape}. '
        f'SparseTensor.replace() is altering sparse structure — geometry would corrupt.'
    )
    print(f'[GATE 1-sparse] PASSED: coords {_captured["frozen"].shape} identical '
          f'(LoRA does not alter sparse geometry).')
    del slat_coord, slat_input, _captured
    gc.collect(); torch.cuda.empty_cache()

    # ── GATE 1-plain: render identity (B=0 → delta=0 → same pixels) ───────────
    slat_raw = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
    raw_color, _ = decode_and_render(pipeline, slat_raw, renderer, diag=False, device=DEVICE)
    raw_render = raw_color.detach().clamp(0, 1)

    slat_lora = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
    with decoder_lora_ctx(dec_model, lora_registry):
        lora_color, _ = decode_and_render(pipeline, slat_lora, renderer, diag=False, device=DEVICE)
    lora_render = lora_color.detach().clamp(0, 1)

    max_diff = (raw_render - lora_render).abs().max().item()
    assert max_diff < 1e-3, f'GATE 1-plain FAIL: max|raw-lora| = {max_diff:.3e} > 1e-3'
    psnr_gate = masked_psnr(lora_render, raw_render, render_mask)
    print(f'[GATE 1-plain] PASSED: max_diff={max_diff:.2e}  '
          f'(init render PSNR vs raw: {psnr_gate:.1f} dB)')
    del raw_render, lora_render
    gc.collect(); torch.cuda.empty_cache()


def gate2_check(lora_registry):
    """GATE 2: check B matrices have nonzero finite gradients after first backward."""
    issues = []
    for b_idx in ACTIVE_BLOCKS:
        lb = lora_registry[b_idx]
        for name, layer in [('qkv', lb.lora_qkv), ('out', lb.lora_out),
                             ('fc1', lb.lora_fc1), ('fc2', lb.lora_fc2)]:
            if layer is None:
                continue
            g = layer.B.grad
            if g is None:
                issues.append(f'b{b_idx}.{name}: B.grad is None')
            elif not torch.isfinite(g).all():
                issues.append(f'b{b_idx}.{name}: B.grad has NaN/Inf')
            elif g.abs().max().item() == 0.0:
                issues.append(f'b{b_idx}.{name}: B.grad is all zeros')
    if issues:
        print('[GATE 2] FAIL:')
        for s in issues:
            print(f'  {s}')
        raise RuntimeError('GATE 2 FAILED — gradients not flowing to decoder LoRA')
    print('[GATE 2] PASSED: all B matrices have finite nonzero gradients.')


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print(f'Rung 5 — Decoder LoRA')
    print(f'  tag          : {_TAG}')
    print(f'  run_id       : {_RUN_ID}')
    print(f'  rank         : {args.rank}')
    print(f'  layers       : {ACTIVE_LAYERS}')
    print(f'  blocks       : {args.blocks} ({ACTIVE_BLOCKS})')
    print(f'  epochs       : {args.epochs}')
    print(f'  seed         : {args.seed}')
    print(f'  expected_params: {EXPECTED_PARAMS:,}')
    print(f'  output       : {_OUT_DIR}')
    print('=' * 70)

    # Save config
    json.dump(
        {**_CFG, 'run_id': _RUN_ID, 'expected_params': EXPECTED_PARAMS,
         'active_blocks': ACTIVE_BLOCKS, 'active_layers': list(ACTIVE_LAYERS)},
        open(_OUT_DIR / 'config.json', 'w'), indent=2
    )

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print(f'\n[SETUP] Loading pipeline from {PRETRAINED}...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']

    # Freeze ALL pipeline parameters
    for p in pipeline.parameters():
        p.requires_grad_(False)
    flow_model.eval()
    dec_model.eval()

    # ── Sparse structure (fixed seed) ─────────────────────────────────────────
    print(f'\n[STRUCT] Sampling structure (seed={STRUCT_SEED})...')
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}')
    del cond_struct
    gc.collect(); torch.cuda.empty_cache()

    # Offload unused models
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except: pass
    torch.cuda.empty_cache()

    # ── Fixed noise ───────────────────────────────────────────────────────────
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'[NOISE] Fixed noise seed={FIXED_SEED}  shape={tuple(fixed_noise_feats.shape)}')

    # ── Encode all frames with frozen DINOv2 (pre-compute once) ───────────────
    print(f'\n[DINO] Encoding {N_FRAMES} frames (frozen, no grad)...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu()
    gc.collect(); torch.cuda.empty_cache()

    # ── Renderer & mask ───────────────────────────────────────────────────────
    renderer    = make_renderer()
    render_mask = load_render_mask()
    n_mask      = render_mask.sum().item()
    print(f'[MASK] {n_mask} masked pixels ({100*n_mask/RENDER_RES**2:.1f}%)')

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    print('[LPIPS] Loading AlexNet...')
    import lpips as _lpips
    lpips_fn = _lpips.LPIPS(net='alex').to(DEVICE); lpips_fn.eval()

    # ── Build decoder LoRA registry ───────────────────────────────────────────
    lora_registry = nn.ModuleDict({
        str(b): DecoderLoRABlock(dim=_DEC_DIM, rank=args.rank, active_layers=ACTIVE_LAYERS)
        for b in ACTIVE_BLOCKS
    })
    lora_registry = lora_registry.to(DEVICE)

    # Helper to access registry by int index
    def get_lb(b_idx):
        return lora_registry[str(b_idx)]
    # Make lora_registry subscriptable by int
    lora_registry.__getitem__ = lambda self, k: self._modules[str(k)]
    type(lora_registry).__getitem__ = lambda self, k: self._modules[str(k)]

    lora_params = list(lora_registry.parameters())
    print(f'[LORA] Registry built:')
    print(f'  active blocks  : {ACTIVE_BLOCKS}')
    print(f'  active layers  : {list(ACTIVE_LAYERS)}')
    print(f'  trainable params: {sum(p.numel() for p in lora_params):,}')

    # ── GATE 0 (runs before checkpoint load so B=0 check is valid) ───────────
    gate0(lora_registry, dec_model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt    = torch.optim.Adam(lora_params, lr=LR, eps=1e-16, weight_decay=0.0)
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
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
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
        gate1(pipeline, flow_model, dec_model, lora_registry, raw_tokens,
              coords, fixed_noise_feats, renderer, render_mask)

    raw_cache   = {}
    frame_list  = TRAIN if not args.smoke else TRAIN[:10]
    gate2_done  = False
    b_norm_ep1  = None  # GATE 3: captured after epoch 1 to verify B grows

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep    = time.time()
        random.shuffle(frame_list)
        n_ok    = 0
        tot_mse = 0.0
        tot_lp  = 0.0
        tot_tot = 0.0

        for frame_i in frame_list:
            opt.zero_grad(set_to_none=True)

            # Step 1: full 25-step denoise (flow model, no grad)
            cond_gl = raw_tokens[frame_i].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
                for t, t_prev in T_PAIRS:
                    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                    v     = flow_model(ns, t_ten, cond_gl)
                    ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)
                slat = normalize_slat(ns)
            del cond_gl, ns

            # Step 2: decode with decoder LoRA ACTIVE (grad flows to LoRA params)
            with decoder_lora_ctx(dec_model, lora_registry):
                color, slat_leaf = decode_and_render(
                    pipeline, slat, renderer, diag=False, device=DEVICE)

            # Step 3: loss
            gt, gt_mask = load_gt(frame_i)
            mse_v, lp_v, loss_v = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)

            (loss_v * loss_scale).backward()
            # slat_leaf.grad is populated but NOT used (no flow backward needed for Rung 5)

            # GATE 2: check after first backward
            if not gate2_done:
                gate2_check(lora_registry)
                gate2_done = True

            # Unscale
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

            del color, slat_leaf, gt, gt_mask, mse_v, lp_v, loss_v, slat
            gc.collect(); torch.cuda.empty_cache()

        avg_mse = tot_mse / max(n_ok, 1)
        avg_lp  = tot_lp  / max(n_ok, 1)
        avg_tot = tot_tot / max(n_ok, 1)

        # Norms
        A_norms, B_norms, BA_norms, eff_ratios, per_block = dec_block_norms(
            lora_registry, dec_model)

        # GATE 3: after epoch 2, verify ||B|| has grown (non-zero update)
        b_norm_now = float(np.mean(B_norms))
        if epoch == 1 and not resumed:
            b_norm_ep1 = b_norm_now
        elif epoch == 2 and b_norm_ep1 is not None:
            assert b_norm_now > b_norm_ep1, (
                f'GATE 3 FAIL: ||B|| did not grow after 2 epochs '
                f'(ep1={b_norm_ep1:.4e}, ep2={b_norm_now:.4e}). '
                f'Decoder LoRA is not updating — check gradient path.'
            )
            print(f'[GATE 3] PASSED: ||B|| grew {b_norm_ep1:.4e} → {b_norm_now:.4e}')

        # Held-out eval
        held_psnr, held_psnr_std, held_lpips, held_ssim, held_res = evaluate_held_out(
            pipeline, flow_model, dec_model, lora_registry,
            raw_tokens, coords, fixed_noise_feats, renderer, render_mask, lpips_fn)

        is_best = held_psnr > best_psnr
        if is_best:
            best_psnr = held_psnr

        _ckpt_data = {
            'run_id': _RUN_ID, 'run': 'rung5', 'epoch': epoch,
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
            save_diagnostics(pipeline, flow_model, dec_model, lora_registry,
                             raw_tokens, coords, fixed_noise_feats, renderer,
                             epoch, raw_cache)
            save_loss_curve(history)
            gc.collect(); torch.cuda.empty_cache()

    # ── Per-block BA_ratio free figure ───────────────────────────────────────
    if history and 'per_block_ratios' in history[-1]:
        last = history[-1]['per_block_ratios']
        blocks = sorted(int(k) for k in last.keys())
        fig, axes = plt.subplots(1, len(ACTIVE_LAYERS), figsize=(5 * len(ACTIVE_LAYERS), 4),
                                 squeeze=False)
        colors = {'qkv': '#98c379', 'out': '#61afef', 'fc1': '#e06c75', 'fc2': '#d19a66'}
        for col, layer_name in enumerate(ACTIVE_LAYERS):
            ax = axes[0][col]
            vals = [last[str(b)].get(layer_name, 0.0) for b in blocks]
            ax.bar(blocks, vals, color=colors.get(layer_name, '#abb2bf'))
            ax.axhline(0.1, color='#e06c75', lw=1, ls='--', label='ceiling (0.1)')
            ax.axhline(0.01, color='#d19a66', lw=1, ls=':', label='floor (0.01)')
            ax.set_xlabel('Block index'); ax.set_ylabel('||B@A|| / ||W||')
            ax.set_title(f'dec.{layer_name}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f'Rung 5 — Per-block decoder correction magnitude (epoch {EPOCHS})')
        fig.tight_layout()
        fig.savefig(_DIAG / 'per_block_BA_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  per-block figure : {_DIAG}/per_block_BA_ratio.png')

    # ── Final 150-frame eval ─────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('[FINAL EVAL] All 150 frames — post-training quality audit')
    print('=' * 70)
    ALL_FRAMES = list(range(1, N_FRAMES + 1))
    final_res  = evaluate_frames(pipeline, flow_model, dec_model, lora_registry,
                                 raw_tokens, coords, fixed_noise_feats,
                                 renderer, render_mask, lpips_fn, ALL_FRAMES)

    held_mask  = [f in set(HELD_OUT) for f in ALL_FRAMES]
    train_mask = [f in set(TRAIN)    for f in ALL_FRAMES]

    def _split(key):
        held  = [r[key] for r, h in zip(final_res['per_frame'], held_mask)  if h]
        train = [r[key] for r, t in zip(final_res['per_frame'], train_mask) if t]
        return held, train

    hp, tp = _split('psnr'); hs, ts = _split('ssim'); hl, tl = _split('lpips')
    final_summary = {
        'all':    {k: final_res[k] for k in
                   ('psnr_mean','psnr_std','ssim_mean','ssim_std','lpips_mean','lpips_std')},
        'held':   dict(psnr_mean=float(np.mean(hp)),  psnr_std=float(np.std(hp)),
                       ssim_mean=float(np.mean(hs)),  ssim_std=float(np.std(hs)),
                       lpips_mean=float(np.mean(hl)), lpips_std=float(np.std(hl))),
        'train':  dict(psnr_mean=float(np.mean(tp)),  psnr_std=float(np.std(tp)),
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
          f'  SSIM={final_summary["held"]["ssim_mean"]:.4f}'
          f'  LPIPS={final_summary["held"]["lpips_mean"]:.4f}')
    print(f'  Train:   PSNR={final_summary["train"]["psnr_mean"]:.3f}±'
          f'{final_summary["train"]["psnr_std"]:.3f} dB')

    print(f'\n[DONE] {EPOCHS} epochs complete.')
    print(f'  best held-out PSNR : {best_psnr:.3f} dB')
    print(f'  config             : {_OUT_DIR}/config.json  (MD5 fragment: {_RUN_ID})')
    print(f'  results            : {_OUT_DIR}')


if __name__ == '__main__':
    main()
