"""
Step 07c — MCFM LoRA Training: real video GT + alpha² regularization

Fixes two bugs in step07b:
  BUG 1: GT was MCFM's own renders (self-consistency) → degenerate supervision,
          Path A already achieves loss≈0, so alpha drifts freely toward 1.
  BUG 2: alpha unbounded → LoRA path (K_pooled) took over, washed out texture.

FIXES:
  FIX 1: GT = real Kling video frames. Path A leaves residual flicker/gap;
          Path B (LoRA) now has a genuine job: close that gap.
  FIX 2: loss += ALPHA_REG * alpha² — penalizes alpha growth, enforces
          "small refinement" intent. alpha grows only if it earns it vs real GT.

Architecture unchanged (dual-path, additive):
  PATH A (frozen) : q_base @ K_hat^T     → out_sp    (MCFM blending, workhorse)
  PATH B (LoRA  ) : q_lora @ K_pooled^T  → out_tp    (temporal refinement)
  out_final       = to_out( out_sp + alpha * out_tp )
  alpha starts at 0.5, regularized to stay small unless data justifies growth.

Evidence logged at every step (for paper):
  - Per-frame: task_loss, reg_loss, total_loss, alpha, alpha_grad
  - Per-epoch: avg_task_loss, avg_reg_loss, alpha, LoRA B-matrix norm
  - Per-epoch: 3-panel strip for frames [1, 75, 150]:
      GT video | MCFM blend (frozen ref) | LoRA render (this epoch)
  - Per-epoch: loss + alpha curve PNG (task_loss, reg_loss, alpha over epochs)
  - Best checkpoint saved separately whenever avg_task_loss improves
  - loss_history.json: full breakdown every epoch

BACKWARD COMPATIBLE:
  - step07b and results_mcfm_{mode}_selfcons_lora_seed6/ are NOT touched
  - New results dir: results_mcfm_{mode}_realgt_lora_seed6/
  - Same LoRA arch, same LOSS_SCALE, same STRUCT_SEED, same FIXED_SEED

Usage:
  python step07c_mcfm_lora_realgt.py --mode v3_C
  python step07c_mcfm_lora_realgt.py --mode v2_C --alpha_reg 0.01
  python step07c_mcfm_lora_realgt.py --mode v3_D --smoke   (3-frame sanity check)
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
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode',       type=str,   default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_pre.add_argument('--epochs',     type=int,   default=50)
_pre.add_argument('--lr',         type=float, default=1e-4)
_pre.add_argument('--rank',       type=int,   default=4)
_pre.add_argument('--loss_scale', type=float, default=4096.0)
_pre.add_argument('--alpha_reg',  type=float, default=0.01,
                  help='L2 regularization on alpha: loss += alpha_reg * alpha^2')
_pre.add_argument('--smoke',      action='store_true',
                  help='Smoke: 1 epoch, frames [1,75,150], no checkpoint saved')
_PRE, _ = _pre.parse_known_args()

if _PRE.smoke:
    _PRE.epochs = 1

_RESULTS  = _HERE / f'results_mcfm_{_PRE.mode}_realgt_lora_seed6'
_CKPT_DIR = _RESULTS / 'lora_ckpts'
_DIAG_DIR = _RESULTS / 'diag_renders'   # per-epoch diagnostic frame 75
_RESULTS.mkdir(parents=True, exist_ok=True)
_CKPT_DIR.mkdir(exist_ok=True)
_DIAG_DIR.mkdir(exist_ok=True)
_LOG_NAME = (f'smoke_{_PRE.mode}.log' if _PRE.smoke
             else f'train_mode{_PRE.mode}_realgt_epochs{_PRE.epochs}.log')


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(_RESULTS / _LOG_NAME)
sys.stderr = sys.stdout

import json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

from step4_mcfm.mcfm        import mcfm_v2, mcfm_v3
from step6_5_lora.lora_v2   import (build_lora_blocks, freeze_trellis,
                                     gate0_verify, trainable_params_v2,
                                     count_trainable_v2)
from step6_5_lora.dual_path_v2 import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render, _gfn,
                                               RENDER_RES, EXTRINSICS, INTRINSICS,
                                               SLAT_MEAN, SLAT_STD)

# ── Constants ──────────────────────────────────────────────────────────────────
# FIX 1: GT = real video frames. DINOv2 conditioning also uses real frames.
VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
GT_FRAMES_DIR = VIDEO_FRAMES_DIR   # real video GT — NOT MCFM renders
GT_FRAME_75   = VIDEO_FRAMES_DIR / 'frame_0075.png'

PRETRAINED  = 'microsoft/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
N_FRAMES    = 150
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
RESCALE_T   = 3.0
GRAD_CLIP   = 1.0
WIRE_PROXY_N = 64

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_window(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(win: list) -> torch.Tensor:
    return torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)


def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt_tensor(frame_idx: int) -> torch.Tensor:
    """Load real video frame as GT. This is what LoRA must learn to match."""
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def blend(mode: str, raw_tokens: dict, frame_idx: int):
    win      = get_window(frame_idx, mode)
    lam      = get_lambda_vec(win)
    tok_d    = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn  = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    stacked  = torch.stack([raw_tokens[i].to(DEVICE) for i in win], dim=0)
    weights  = torch.tensor([2.0 if i == frame_idx else 1.0 for i in win],
                             dtype=torch.float32, device=DEVICE)
    weights  = weights / weights.sum()
    K_pooled = (weights[:, None, None] * stacked).sum(0)
    return K_hat, K_pooled, win


def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled, lora_blocks, alpha, require_grad: bool):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx   = dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None)
    if require_grad:
        with ctx:
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with ctx:
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


def find_latest_ckpt():
    ckpts = sorted(_CKPT_DIR.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


DIAG_KEYFRAMES = [1, 75, 150]   # frames rendered every epoch for visual evidence
_FONT = None

def _get_font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
        except Exception:
            _FONT = ImageFont.load_default()
    return _FONT


def render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                        raw_tokens, mode, frame_idx, coords,
                        fixed_noise_feats, renderer):
    """Render a single frame with current LoRA weights. Returns (3,H,W) float32 tensor."""
    K_hat, K_pooled, _ = blend(mode, raw_tokens, frame_idx)
    ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl = K_hat.unsqueeze(0)
    # Denoise with flow_model on GPU, then move it off to free VRAM for decoder
    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled.to(DEVICE), lora_blocks,
                                   alpha, enhance_bias=None):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    flow_model.cpu()
    gc.collect(); torch.cuda.empty_cache()
    slat = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    flow_model.to(DEVICE)
    return color.detach().clamp(0, 1)   # (3, H, W)


def render_mcfm_baseline(pipeline, raw_tokens, mode, frame_idx,
                          coords, fixed_noise_feats, renderer, flow_model):
    """Render frame using frozen TRELLIS + MCFM conditioning, zero LoRA (alpha=0)."""
    K_hat, K_pooled, _ = blend(mode, raw_tokens, frame_idx)
    ns  = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cgl = K_hat.unsqueeze(0)
    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(ns, t_ten, cgl)   # frozen only, no dual-path needed at alpha=0
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    flow_model.cpu()
    gc.collect(); torch.cuda.empty_cache()
    slat  = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    flow_model.to(DEVICE)
    return color.detach().clamp(0, 1)


def make_3panel_strip(gt_path: Path, mcfm_tensor, lora_tensor,
                      epoch: int, frame_idx: int, cell: int = 320) -> Image.Image:
    """
    3-panel strip: GT video | MCFM blend (frozen ref) | LoRA render (epoch N)
    Labels on top. Returns PIL Image (cell*3, cell+30).
    """
    label_h = 28
    labels  = ['GT video', 'MCFM blend (ref)', f'LoRA e{epoch:03d}']
    imgs = [
        Image.open(gt_path).convert('RGB').resize((cell, cell), Image.LANCZOS),
        Image.fromarray(
            (mcfm_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        ).resize((cell, cell), Image.LANCZOS),
        Image.fromarray(
            (lora_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        ).resize((cell, cell), Image.LANCZOS),
    ]
    canvas = Image.new('RGB', (cell * 3, cell + label_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    font   = _get_font()
    for ci, (img, lbl) in enumerate(zip(imgs, labels)):
        canvas.paste(img, (ci * cell, label_h))
        draw.rectangle([ci * cell, 0, (ci+1)*cell - 1, label_h - 1], fill=(35, 35, 50))
        try:
            bbox = draw.textbbox((0,0), lbl, font=font)
            tw   = bbox[2] - bbox[0]
        except AttributeError:
            tw = len(lbl) * 8
        draw.text((ci * cell + (cell - tw)//2, 7), lbl, fill=(220,220,220), font=font)
    draw.text((cell*3 - 65, cell + label_h - 16), f'f{frame_idx:03d}',
              fill=(140,140,140), font=font)
    return canvas


def save_epoch_diagnostics(pipeline, flow_model, lora_blocks, alpha,
                            raw_tokens, mode, coords, fixed_noise_feats,
                            renderer, epoch: int, mcfm_cache: dict):
    """
    Run after every epoch:
      1. Render frames [1, 75, 150] with current LoRA
      2. Save 3-panel strip: GT | MCFM | LoRA for each frame
      3. Save combined 3×3 grid (rows=keyframes, cols=GT/MCFM/LoRA)
    """
    epoch_diag_dir = _DIAG_DIR / f'e{epoch:03d}'
    epoch_diag_dir.mkdir(exist_ok=True)
    font = _get_font()

    strips = []
    for fi in DIAG_KEYFRAMES:
        print(f'  [DIAG] rendering frame {fi}...')
        lora_t = render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                                      raw_tokens, mode, fi, coords,
                                      fixed_noise_feats, renderer)
        # Get or compute MCFM baseline (frozen once, cached)
        if fi not in mcfm_cache:
            mcfm_cache[fi] = render_mcfm_baseline(
                pipeline, raw_tokens, mode, fi,
                coords, fixed_noise_feats, renderer, flow_model)

        gt_path = GT_FRAMES_DIR / f'frame_{fi:04d}.png'
        strip   = make_3panel_strip(gt_path, mcfm_cache[fi], lora_t, epoch, fi)
        strip.save(epoch_diag_dir / f'strip_f{fi:04d}.png')
        strips.append(strip)

        # Also save just the LoRA render alone
        arr = (lora_t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_diag_dir / f'lora_f{fi:04d}.png')

    # Combined 3×3 grid
    if strips:
        cell   = 320
        n_col  = 3   # GT | MCFM | LoRA
        n_row  = len(DIAG_KEYFRAMES)
        label_h = 28
        grid = Image.new('RGB', (cell * n_col, (cell + label_h) * n_row), (10, 10, 10))
        for ri, strip in enumerate(strips):
            grid.paste(strip, (0, ri * (cell + label_h)))
        grid.save(_DIAG_DIR / f'grid_e{epoch:03d}.png')
        print(f'  [DIAG] saved grid_e{epoch:03d}.png  (3×3: keyframes × [GT|MCFM|LoRA])')


def save_loss_curve(loss_history: list):
    """Save task_loss, reg_loss, alpha curve as PNG. Called every epoch."""
    if len(loss_history) < 1:
        return
    epochs    = [r['epoch']         for r in loss_history]
    task_loss = [r['avg_task_loss'] for r in loss_history]
    reg_loss  = [r['avg_reg_loss']  for r in loss_history]
    alphas    = [r['alpha']         for r in loss_history]
    ag        = [r.get('alpha_grad_mean', float('nan')) for r in loss_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training diagnostics — step07c (real GT + alpha² reg)', fontsize=11)

    axes[0].plot(epochs, task_loss, 'o-', color='#e06c75', lw=2, ms=5, label='task_loss')
    axes[0].plot(epochs, reg_loss,  's--', color='#61afef', lw=1.5, ms=4, label='reg_loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Task loss + Reg loss')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, alphas, 'o-', color='#98c379', lw=2, ms=5)
    axes[1].axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='init=0.5')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('alpha')
    axes[1].set_title('alpha over epochs\n(should stay ≤ 0.5)')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    valid_ag = [(e, g) for e, g in zip(epochs, ag) if not np.isnan(g)]
    if valid_ag:
        e_ag, g_ag = zip(*valid_ag)
        axes[2].bar(e_ag, g_ag,
                    color=['#e06c75' if g > 0 else '#98c379' for g in g_ag],
                    alpha=0.8)
        axes[2].axhline(0, color='black', lw=0.8)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('alpha gradient mean')
        axes[2].set_title('Alpha gradient\n(negative = reg pushing alpha down ✓)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(_DIAG_DIR / 'loss_curve.png', dpi=130, bbox_inches='tight')
    plt.close()


def lora_b_norm(lora_blocks) -> float:
    """RMS norm of all LoRA B matrices. Should grow as LoRA learns."""
    vals = [p.float().norm().item()
            for blk in lora_blocks
            for name, p in blk.named_parameters() if 'B' in name]
    return float(np.sqrt(np.mean([v**2 for v in vals]))) if vals else 0.0


def wire_check(mode, raw_tokens, flow_model, lora_blocks, alpha, coords, fixed_noise_feats):
    print('\n[WIRE CHECK] Running on frame 75 (no enhancement)...')
    K_hat, K_pooled, win = blend(mode, raw_tokens, 75)
    tok_curr = raw_tokens[75].to(DEVICE)

    print(f'  [W0-SHAPE ] K_hat={tuple(K_hat.shape)}  K_pooled={tuple(K_pooled.shape)}  '
          f'window={win}  mode={mode}')

    diff = (K_hat.float() - tok_curr.float()).abs()
    print(f'  [W1-MCFM  ] K_hat vs raw tok[75]: diff_mean={diff.mean():.6f}  '
          f'diff_max={diff.max():.6f}  '
          f'{"blending active ✓" if diff.max() > 1e-6 else "IDENTICAL ← no blend"}')

    expected_pool = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0).mean(0)
    pool_err = (K_pooled.float() - expected_pool.float()).abs().max().item()
    print(f'  [W2-KPOOL ] pool_err={pool_err:.2e}  {"✓" if pool_err < 1e-5 else "MISMATCH"}')

    blk0     = flow_model.blocks[0].cross_attn
    blk_lora = lora_blocks[0]
    with torch.no_grad():
        dummy_x = torch.randn(WIRE_PROXY_N, blk0.channels, device=DEVICE,
                              dtype=next(blk_lora.parameters()).dtype)
        q_lora  = blk_lora.lora_q.forward(dummy_x)
        kv_lora = blk_lora.lora_kv.forward(
            K_pooled[:WIRE_PROXY_N].to(next(blk_lora.parameters()).dtype))
    B_zero = all(p.abs().max().item() < 1e-9
                 for blk in lora_blocks for name, p in blk.named_parameters() if 'B' in name)
    print(f'  [W3-LORA  ] q_max={q_lora.float().abs().max():.4f}  '
          f'kv_max={kv_lora.float().abs().max():.4f}  alpha={alpha.item():.4f}  '
          f'B_all_zero={B_zero}  {"B=0 at init ✓" if B_zero else "B nonzero (resumed)"}')

    noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl  = K_hat.unsqueeze(0)
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:2]:
            t_ten    = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v_frozen = flow_model(noise_sp, t_ten, cond_gl)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None):
                v_dual = flow_model(noise_sp, t_ten, cond_gl)
            diff4 = (v_frozen.feats.float() - v_dual.feats.float()).abs()
            print(f'  [W4-DUAL  ] t={t:.3f}: max={diff4.max():.6f}  '
                  f'{"paths differ ✓" if diff4.max() > 1e-8 else "IDENTICAL ← dual-path not firing"}')
            noise_sp = noise_sp.replace(noise_sp.feats - (t - t_prev) * v_frozen.feats)
    print('[WIRE CHECK] Done.\n')


def gate1_backward_compat(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, coords):
    print('\n[GATE 1] Backward compat (alpha=0, B=zeros, no enhancement)...')
    alpha_zero = nn.Parameter(torch.tensor(0.0, device=DEVICE))

    def _run(use_lora):
        x = sp.SparseTensor(feats=noise_sp.feats.clone(), coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                if use_lora:
                    with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks,
                                          alpha_zero, enhance_bias=None):
                        v = flow_model(x, t_ten, cond_gl)
                else:
                    v = flow_model(x, t_ten, cond_gl)
                x = x.replace(x.feats - (t - t_prev) * v.feats)
        return x

    x_base = _run(False); x_lora = _run(True)
    diff   = (x_base.feats.float() - x_lora.feats.float()).abs()
    tol    = 1e-2
    print(f'  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}  tol={tol}')
    if diff.max() < tol:
        print('  [GATE 1] PASSED ✓  (LoRA clean at alpha=0)')
    else:
        raise RuntimeError(f'GATE 1 FAILED: max_diff={diff.max():.6f}')


def gate2_grad_check(lora_blocks, alpha):
    print('\n[GATE 2] Gradient check...')
    ag = alpha.grad
    print(f'  alpha={alpha.item():.6f}  alpha.grad='
          f'{"NONE ← FAIL" if ag is None else f"{ag.item():.6e}  "}'
          f'{"(pushing alpha UP ← check reg)" if ag is not None and ag.item() > 0 else "(pushing alpha DOWN ✓)" if ag is not None else ""}')

    q_A, q_B, kv_A, kv_B = [], [], [], []
    q_fail = kv_fail = 0
    for i, blk in enumerate(lora_blocks):
        for name, p in [('lora_q.A', blk.lora_q.A), ('lora_q.B', blk.lora_q.B),
                        ('lora_kv.A', blk.lora_kv.A), ('lora_kv.B', blk.lora_kv.B)]:
            if p.grad is None or p.grad.abs().max().item() == 0:
                print(f'  blk{i:02d} {name}: ZERO GRAD ← FAIL')
                if 'q.' in name: q_fail += 1
                else: kv_fail += 1
            mx = p.grad.abs().max().item() if p.grad is not None else 0.0
            if   'q.A'  in name: q_A.append(mx)
            elif 'q.B'  in name: q_B.append(mx)
            elif 'kv.A' in name: kv_A.append(mx)
            elif 'kv.B' in name: kv_B.append(mx)

    for vals, nm in [(q_A,'lora_q  A'), (q_B,'lora_q  B'),
                     (kv_A,'lora_kv A'), (kv_B,'lora_kv B')]:
        if vals:
            print(f'  {nm}: min={min(vals):.3e}  max={max(vals):.3e}  '
                  f'mean={sum(vals)/len(vals):.3e}')

    frozen_nonzero = sum(1 for _, p in lora_blocks.named_parameters()
                         if not p.requires_grad and p.grad is not None
                         and p.grad.abs().max() > 0)
    print(f'  frozen with nonzero grad: {frozen_nonzero}  (expected 0)')
    if q_fail == 0 and kv_fail == 0 and ag is not None and frozen_nonzero == 0:
        print('  [GATE 2] PASSED ✓')
    else:
        print(f'  [GATE 2] issues: q_fail={q_fail} kv_fail={kv_fail} '
              f'frozen_nonzero={frozen_nonzero}')


def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode',         type=str,   default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--rank',         type=int,   default=4)
    parser.add_argument('--loss_scale',   type=float, default=4096.0)
    parser.add_argument('--alpha_reg',    type=float, default=0.01,
                        help='L2 regularization weight on alpha: loss += alpha_reg * alpha^2')
    parser.add_argument('--frame_stride', type=int,   default=1)
    parser.add_argument('--smoke',        action='store_true')
    parser.add_argument('--diag_every',   type=int,   default=1,
                        help='Save 3-panel diagnostic strips every N epochs (default: every epoch)')
    args = parser.parse_args()

    mode         = args.mode
    EPOCHS       = args.epochs
    LR           = args.lr
    LORA_RANK    = args.rank
    LOSS_SCALE   = args.loss_scale
    ALPHA_REG    = args.alpha_reg
    FRAME_STRIDE = args.frame_stride
    SMOKE        = args.smoke
    DIAG_EVERY   = args.diag_every

    print('=' * 72)
    print('Step 07c — MCFM LoRA Training: real video GT + alpha² regularization')
    print('=' * 72)
    print(f'  mode        : {mode}')
    print(f'  GT source   : REAL VIDEO FRAMES ({GT_FRAMES_DIR})')
    print(f'  alpha_reg   : {ALPHA_REG}  (L2 penalty on alpha — FIX 2)')
    print(f'  epochs      : {EPOCHS}')
    print(f'  lr          : {LR}')
    print(f'  lora_rank   : {LORA_RANK}')
    print(f'  loss_scale  : {LOSS_SCALE}  (fp16 FTZ fix)')
    print(f'  fixed_seed  : {FIXED_SEED}')
    print(f'  struct_seed : {STRUCT_SEED}')
    print(f'  diag_every  : {DIAG_EVERY} epochs  (frame 75 saved to {_DIAG_DIR})')
    print(f'  results     : {_RESULTS}')
    print(f'  ckpt_dir    : {_CKPT_DIR}')
    print(f'  log         : {_RESULTS / _LOG_NAME}')

    # ── Verify GT frames exist ─────────────────────────────────────────────────
    missing = [i for i in [1, 75, 150]
               if not (GT_FRAMES_DIR / f'frame_{i:04d}.png').exists()]
    if missing:
        raise FileNotFoundError(f'GT frames missing: {missing} in {GT_FRAMES_DIR}')
    print(f'\n[GT CHECK] Spot-checked frames [1, 75, 150] in {GT_FRAMES_DIR} ✓')

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('\n[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Voxel structure ────────────────────────────────────────────────────────
    print(f'\n[STRUCT] Sampling structure (STRUCT_SEED={STRUCT_SEED}, frame 75)...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}  coords_sum={coords.sum().item()}  (expect 7301, 661358)')
    assert N_vox == 7301, f'N_vox={N_vox} != 7301 — structure mismatch'

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA setup ─────────────────────────────────────────────────────────────
    print('\n[LORA] Building LoRA blocks...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, n_blocks=24).to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))
    gate0_verify(lora_blocks, alpha)
    n_trainable = count_trainable_v2(lora_blocks, alpha)
    n_frozen    = sum(p.numel() for p in flow_model.parameters())
    print(f'  trainable={n_trainable:,}  frozen={n_frozen:,}')

    # ── Optimizer + resume ─────────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
    start_epoch  = 1
    loss_history = []
    best_task_loss = float('inf')
    latest_ckpt  = find_latest_ckpt()

    if latest_ckpt:
        print(f'\n[RESUME] Loading {latest_ckpt.name}...')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        if 'mode' in ckpt and ckpt['mode'] != mode:
            raise RuntimeError(f'Checkpoint mode={ckpt["mode"]} != --mode={mode}')
        if 'beta' in ckpt:
            raise RuntimeError('Checkpoint has beta key — wrong script (step07). Delete ckpts.')
        if ckpt.get('gt_source') != 'real_video':
            raise RuntimeError(
                f'Checkpoint gt_source={ckpt.get("gt_source")} — '
                f'this is a selfcons checkpoint (step07b). '
                f'Delete {_CKPT_DIR} to start fresh with real GT.')
        print(f'  [RESUME-CHECK] mode={mode} ✓  gt_source=real_video ✓')
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha     = nn.Parameter(ckpt['alpha'].to(DEVICE))
        optimizer = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch    = ckpt['epoch'] + 1
        best_task_loss = ckpt.get('best_task_loss', float('inf'))
        hist_path      = _RESULTS / 'loss_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                loss_history = json.load(f)
        print(f'  epoch={ckpt["epoch"]}  avg_task_loss={ckpt["avg_task_loss"]:.5f}  '
              f'avg_reg_loss={ckpt.get("avg_reg_loss", float("nan")):.5f}  '
              f'alpha={alpha.item():.4f}  best_task_loss={best_task_loss:.5f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs done. Nothing to train.')
        return

    # ── Fixed noise ────────────────────────────────────────────────────────────
    print(f'\n[NOISE] Building fixed noise (seed={FIXED_SEED})...')
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'  shape={tuple(fixed_noise_feats.shape)}  '
          f'mean={fixed_noise_feats.mean():.6f}  std={fixed_noise_feats.std():.6f}')

    # ── DINOv2 encode all 150 frames ──────────────────────────────────────────
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    t_enc = time.time()
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            t = raw_tokens[i].float()
            print(f'  {i}/{N_FRAMES}  mean={t.mean():.5f}  std={t.std():.5f}')
    dino_model.cpu(); torch.cuda.empty_cache()
    print(f'  Done in {time.time()-t_enc:.1f}s')

    # ── K_pooled cache ─────────────────────────────────────────────────────────
    print(f'\n[CACHE] Building K_pooled cache ({N_FRAMES} frames)...')
    kpooled_cache = {}
    for fi in range(1, N_FRAMES + 1):
        win = get_window(fi, mode)
        kpooled_cache[fi] = torch.stack([raw_tokens[i] for i in win], 0).mean(0).cpu()
    print(f'  Done. {len(kpooled_cache)} entries.')

    # ── Renderer ───────────────────────────────────────────────────────────────
    renderer = make_renderer()

    # ── Gate 1 — backward compat ───────────────────────────────────────────────
    K_hat_75, Kp_75, _ = blend(mode, raw_tokens, 75)
    noise_sp_g1 = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    gate1_backward_compat(flow_model, noise_sp_g1, K_hat_75.unsqueeze(0),
                          Kp_75, lora_blocks, coords)
    del noise_sp_g1, K_hat_75, Kp_75; torch.cuda.empty_cache()

    # ── Wire check ─────────────────────────────────────────────────────────────
    wire_check(mode, raw_tokens, flow_model, lora_blocks, alpha,
               coords, fixed_noise_feats)

    # ── Training loop ──────────────────────────────────────────────────────────
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.train()
    frame_list  = [1, 75, 150] if SMOKE else list(range(1, N_FRAMES + 1, FRAME_STRIDE))

    print(f'\n[TRAIN] Epochs {start_epoch}→{EPOCHS}  '
          f'frames/epoch={len(frame_list)}  STEPS={STEPS}  '
          f'LOSS_SCALE={LOSS_SCALE}  ALPHA_REG={ALPHA_REG}  mode={mode}')
    print(f'  GT source: REAL VIDEO FRAMES  (FIX 1)')
    print(f'  alpha²  regularization: {ALPHA_REG} * alpha²  (FIX 2)')

    gate2_done  = False
    first_diag  = True
    mcfm_cache  = {}   # cache MCFM baseline renders (fixed, computed once per frame)

    print(f'\n[DIAG] Keyframes rendered every epoch: {DIAG_KEYFRAMES}')
    print(f'       3-panel strips → {_DIAG_DIR}/e{{epoch:03d}}/strip_f{{fi:04d}}.png')
    print(f'       Combined grid  → {_DIAG_DIR}/grid_e{{epoch:03d}}.png')
    print(f'       Loss curve     → {_DIAG_DIR}/loss_curve.png')

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_task_loss = 0.0
        epoch_reg_loss  = 0.0
        epoch_alpha_grads = []
        t_epoch = time.time()

        for frame_i in frame_list:
            K_hat, K_pooled, win = blend(mode, raw_tokens, frame_i)
            cond_gl  = K_hat.unsqueeze(0)
            noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)

            optimizer.zero_grad()

            # 24-step prefix — no grad
            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha)

            # Step 25 val pass — no grad, get slat gradient via decode+render
            x0_val   = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha,
                require_grad=False)
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp

            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None: _flow_ref.cpu()
            gc.collect(); torch.cuda.empty_cache()

            try:
                color, slat_leaf_feats = decode_and_render(
                    pipeline, slat_val, renderer, diag=first_diag, device=DEVICE)
                gc.collect(); torch.cuda.empty_cache()

                gt        = load_gt_tensor(frame_i)
                task_loss = F.mse_loss(color, gt)

                if first_diag:
                    print(f'  [G4] task_loss={_gfn(task_loss)}  val={task_loss.item():.5f}')

                # Scale task loss for fp16 FTZ fix
                (task_loss * LOSS_SCALE).backward()
                if slat_leaf_feats.grad is not None:
                    slat_leaf_feats.grad.div_(LOSS_SCALE)

                if first_diag:
                    g = slat_leaf_feats.grad
                    print(f'  [G5] slat_leaf_feats.grad: '
                          f'{"None" if g is None else f"dtype={g.dtype} max={g.abs().max():.3e}"}')
                    first_diag = False

            finally:
                if _flow_ref is not None: _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render: no gradient on slat_leaf_feats')
            grad_slat_max = slat_leaf_feats.grad.abs().max().item()
            if grad_slat_max < 1e-30:
                print(f'  WARNING e{epoch:02d} f{frame_i:03d}: slat grad ZERO')

            # Step 25 train pass — builds grad graph through LoRA
            grad_slat = slat_leaf_feats.grad.detach()
            x0_raw    = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha,
                require_grad=True)
            slat = normalize_slat(x0_raw)
            torch.autograd.backward(slat.feats, grad_slat)

            # FIX 2: alpha² regularization — adds 2*ALPHA_REG*alpha to alpha.grad
            reg_loss = ALPHA_REG * alpha ** 2
            reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params_v2(lora_blocks, alpha), GRAD_CLIP)

            # Record alpha gradient BEFORE optimizer step (evidence)
            alpha_grad_val = alpha.grad.item() if alpha.grad is not None else float('nan')
            epoch_alpha_grads.append(alpha_grad_val)

            optimizer.step()

            if not gate2_done:
                gate2_grad_check(lora_blocks, alpha)
                gate2_done = True

            task_loss_val = task_loss.item()
            reg_loss_val  = reg_loss.item()
            epoch_task_loss += task_loss_val
            epoch_reg_loss  += reg_loss_val

            if frame_i == frame_list[0] or frame_i % 10 == 0:
                print(f'  e{epoch:02d} f{frame_i:03d}  '
                      f'task={task_loss_val:.5f}  reg={reg_loss_val:.5f}  '
                      f'total={task_loss_val+reg_loss_val:.5f}  '
                      f'alpha={alpha.item():.4f}  alpha_grad={alpha_grad_val:.3e}  '
                      f'win={win}')

            del color, slat_leaf_feats, gt, task_loss, reg_loss
            del x0_raw, slat, grad_slat, x_prefix, slat_val, cond_gl, K_pooled, K_hat
            gc.collect(); torch.cuda.empty_cache()

        # ── Epoch summary ──────────────────────────────────────────────────────
        avg_task  = epoch_task_loss / len(frame_list)
        avg_reg   = epoch_reg_loss  / len(frame_list)
        avg_ag    = float(np.mean([g for g in epoch_alpha_grads if not np.isnan(g)]))
        b_norm    = lora_b_norm(lora_blocks)
        elapsed   = time.time() - t_epoch
        is_best   = avg_task < best_task_loss
        if is_best:
            best_task_loss = avg_task

        print(f'[EPOCH {epoch:02d}] avg_task={avg_task:.5f}  avg_reg={avg_reg:.5f}  '
              f'total={avg_task+avg_reg:.5f}  alpha={alpha.item():.4f}  '
              f'alpha_grad_mean={avg_ag:.3e}  lora_B_norm={b_norm:.4f}  '
              f'{"BEST ✓" if is_best else ""}  time={elapsed:.1f}s')

        if avg_ag > 0:
            print(f'  [ALPHA WARNING] alpha gradient POSITIVE ({avg_ag:.3e}) — '
                  f'still drifting UP. Increase --alpha_reg.')
        else:
            print(f'  [ALPHA OK] alpha gradient negative ({avg_ag:.3e}) — '
                  f'reg holding alpha down ✓')

        loss_history.append({
            'epoch': epoch, 'avg_task_loss': avg_task, 'avg_reg_loss': avg_reg,
            'avg_total_loss': avg_task + avg_reg, 'alpha': alpha.item(),
            'alpha_grad_mean': avg_ag, 'lora_B_norm': b_norm, 'is_best': is_best,
        })
        with open(_RESULTS / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

        # ── Save checkpoint ────────────────────────────────────────────────────
        if SMOKE:
            print('  [CKPT] smoke mode — skipping checkpoint save')
        else:
            ckpt_data = {
                'epoch'          : epoch,
                'mode'           : mode,
                'gt_source'      : 'real_video',   # guard against selfcons ckpts
                'alpha_reg'      : ALPHA_REG,
                'lora_state'     : lora_blocks.state_dict(),
                'alpha'          : alpha.data,
                'optimizer'      : optimizer.state_dict(),
                'avg_task_loss'  : avg_task,
                'avg_reg_loss'   : avg_reg,
                'best_task_loss' : best_task_loss,
            }
            ckpt_path = _CKPT_DIR / f'lora_e{epoch:03d}.pt'
            torch.save(ckpt_data, ckpt_path)
            print(f'  [CKPT] saved {ckpt_path.name}')

            if is_best:
                best_path = _CKPT_DIR / 'lora_best.pt'
                torch.save(ckpt_data, best_path)
                print(f'  [CKPT] new best → lora_best.pt  (avg_task={avg_task:.5f})')

        # ── Per-epoch diagnostics: 3-panel strips + loss curve ────────────────
        do_diag = (epoch % DIAG_EVERY == 0 or epoch == EPOCHS) and not SMOKE
        if do_diag:
            print(f'  [DIAG] Running epoch {epoch} visual diagnostics (3 keyframes)...')
            flow_model.eval()
            save_epoch_diagnostics(pipeline, flow_model, lora_blocks, alpha,
                                   raw_tokens, mode, coords, fixed_noise_feats,
                                   renderer, epoch, mcfm_cache)
            save_loss_curve(loss_history)
            flow_model.train()
            gc.collect(); torch.cuda.empty_cache()

        gc.collect(); torch.cuda.empty_cache()

    print(f'\n[DONE] Training complete. {EPOCHS} epochs.')
    print(f'  best_task_loss = {best_task_loss:.5f}  (lora_best.pt)')
    print(f'  Results: {_RESULTS}')
    print(f'  Diagnostic renders: {_DIAG_DIR}')


if __name__ == '__main__':
    main()
