"""
Step 07b — MCFM LoRA Training (no enhancement)

Trains LoRA weights on the SLaT flow model with MCFM-blended conditioning.
NO enhancement bias — pure temporal blending (v2 or v3) + LoRA on top.

BACKWARD COMPATIBLE:
  - step07_mcfm_lora_train.py and its checkpoints are NOT touched
  - results_mcfm_{mode}_lora_seed6/ is NOT touched
  - Same LoRA architecture : lora_v2 + dual_path_v2
  - Same fixed noise       : seed=6, STRUCT_SEED=42, LOSS_SCALE=4096

Architecture (dual-path cross-attention, enhance_bias=None throughout):
  PATH A (frozen) : q_base @ K_hat^T            → out_sp
  PATH B (LoRA  ) : q_lora @ K_pooled^T         → out_tp
  out_final       = to_out( out_sp + alpha * out_tp )

K_hat    = MCFM blend of window tokens (v2 or v3)
K_pooled = simple mean of window tokens (PATH B input)

Results : results_mcfm_{mode}_noenh_lora_seed6/

Usage:
  python step07b_mcfm_lora_noenh.py --mode v2_C
  python step07b_mcfm_lora_noenh.py --mode v2_D --epochs 50
  python step07b_mcfm_lora_noenh.py --mode v3_C --lr 1e-4
  python step07b_mcfm_lora_noenh.py --mode v3_D --rank 4  # auto-resumes if ckpt exists
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

# ── Early arg parse — must happen before Tee ──────────────────────────────────
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode',       type=str,  default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_pre.add_argument('--epochs',     type=int,  default=50)
_pre.add_argument('--lr',         type=float, default=1e-4)
_pre.add_argument('--rank',       type=int,  default=4)
_pre.add_argument('--loss_scale', type=float, default=4096.0)
_pre.add_argument('--smoke',      action='store_true',
                  help='Smoke test: 1 epoch, frames [1,75,150] only, no checkpoint saved')
_PRE, _ = _pre.parse_known_args()

if _PRE.smoke:
    _PRE.epochs = 1

_RESULTS = _HERE / f'results_mcfm_{_PRE.mode}_selfcons_lora_seed6'
_RESULTS.mkdir(parents=True, exist_ok=True)
_CKPT_DIR = _RESULTS / 'lora_ckpts'
_CKPT_DIR.mkdir(exist_ok=True)
_LOG_NAME = f'smoke_{_PRE.mode}.log' if _PRE.smoke else f'train_mode{_PRE.mode}_noenh_epochs{_PRE.epochs}.log'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)   # append so resume doesn't wipe log
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()


sys.stdout = _Tee(_RESULTS / _LOG_NAME)
sys.stderr = sys.stdout

# ── Imports after Tee ─────────────────────────────────────────────────────────
import json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

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
# Original video frames — used ONLY for DINOv2 conditioning and structure sampling
VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
GT_FRAME_75 = VIDEO_FRAMES_DIR / 'frame_0075.png'

# MCFM rendered frames — used as GT for LoRA loss (self-consistency training)
# The LoRA learns to reproduce what MCFM blending already achieves
GT_FRAMES_DIR = (Path(__file__).resolve().parent
                 / f'results_mcfm_{_PRE.mode}_seed6_fixednoise' / 'beta0p0')
PRETRAINED    = 'microsoft/TRELLIS-image-large'
DEVICE        = torch.device('cuda')
N_FRAMES      = 150
STRUCT_SEED   = 42
FIXED_SEED    = 6
STEPS         = 25
RESCALE_T     = 3.0
GRAD_CLIP     = 1.0
WIRE_PROXY_N  = 64   # number of proxy voxels used in wire check (no artifact needed)

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── Window helpers ─────────────────────────────────────────────────────────────

def get_window(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    else:
        return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(win: list) -> torch.Tensor:
    return torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)


# ── DINOv2 encoding ────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)   # (1374, 1024)


def load_gt_tensor(frame_idx: int) -> torch.Tensor:
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


# ── MCFM blend ─────────────────────────────────────────────────────────────────

def blend(mode: str, raw_tokens: dict, frame_idx: int):
    """Returns (K_hat, K_pooled, win).
    K_hat    = MCFM blend of window tokens  (1374, 1024)
    K_pooled = simple mean of window tokens (1374, 1024)  — PATH B input
    """
    win      = get_window(frame_idx, mode)
    lam      = get_lambda_vec(win)
    tok_d    = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn  = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    K_pooled = torch.stack([raw_tokens[i].to(DEVICE) for i in win], dim=0).mean(0)
    return K_hat, K_pooled, win


# ── Denoise helpers ────────────────────────────────────────────────────────────
# enhance_bias is always None in this script.

def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled,
                           lora_blocks, alpha):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                                   enhance_bias=None):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled,
                      lora_blocks, alpha, require_grad: bool):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx   = dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                              enhance_bias=None)
    if require_grad:
        with ctx:
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with ctx:
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def find_latest_ckpt():
    ckpts = sorted(_CKPT_DIR.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Wire check ─────────────────────────────────────────────────────────────────

def wire_check(mode: str, raw_tokens: dict, flow_model,
               lora_blocks, alpha, coords, fixed_noise_feats):
    """
    Startup wire check. No enhancement — only MCFM + LoRA wires.

    W1 — MCFM blend is active (K_hat != raw single frame token)
    W2 — K_pooled is correct mean of window tokens
    W3 — LoRA PATH B fires (q_lora / kv_lora produce nonzero output)
    W4 — Dual-path output differs from frozen at alpha=0.5
    """
    print('\n[WIRE CHECK] Running on frame 75 (no enhancement)...')

    K_hat, K_pooled, win = blend(mode, raw_tokens, 75)
    tok_curr = raw_tokens[75].to(DEVICE)

    # ── W0: shapes ────────────────────────────────────────────────────────────
    print(f'  [W0-SHAPE ] K_hat={tuple(K_hat.shape)}  K_pooled={tuple(K_pooled.shape)}  '
          f'window={win}  mode={mode}')

    # ── W1: MCFM blend is nontrivial ─────────────────────────────────────────
    diff = (K_hat.float() - tok_curr.float()).abs()
    blend_active = diff.max().item() > 1e-6
    print(f'  [W1-MCFM  ] K_hat mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
    print(f'  [W1-MCFM  ] K_hat vs raw tok[75]: diff_mean={diff.mean():.6f}  '
          f'diff_max={diff.max():.6f}  '
          f'{"blending active ✓" if blend_active else "IDENTICAL ← no blend (degenerate)"}')

    # ── W2: K_pooled is correct mean ─────────────────────────────────────────
    expected_pool = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0).mean(0)
    pool_err = (K_pooled.float() - expected_pool.float()).abs().max().item()
    print(f'  [W2-KPOOL ] K_pooled mean={K_pooled.float().mean():.5f}  '
          f'std={K_pooled.float().std():.5f}  '
          f'pool_err={pool_err:.2e}  {"✓" if pool_err < 1e-5 else "MISMATCH ←"}')

    # ── W3: LoRA PATH B fires ─────────────────────────────────────────────────
    blk0 = flow_model.blocks[0].cross_attn
    with torch.no_grad():
        blk_lora = lora_blocks[0]
        dummy_x  = torch.randn(WIRE_PROXY_N, blk0.channels, device=DEVICE,
                               dtype=next(blk_lora.parameters()).dtype)
        q_lora   = blk_lora.lora_q.forward(dummy_x)
        kv_lora  = blk_lora.lora_kv.forward(
            K_pooled[:WIRE_PROXY_N].to(next(blk_lora.parameters()).dtype)
        )
        q_norm  = q_lora.float().abs().max().item()
        kv_norm = kv_lora.float().abs().max().item()
    B_zero = all(
        p.abs().max().item() < 1e-9
        for blk in lora_blocks
        for name, p in blk.named_parameters() if 'B' in name
    )
    print(f'  [W3-LORA  ] lora_q out max={q_norm:.4f}  lora_kv out max={kv_norm:.4f}  '
          f'alpha={alpha.item():.4f}  B_all_zero={B_zero}  '
          f'{"B=0 at init ✓" if B_zero else "B nonzero (resumed or bug) ←"}')

    # ── W4: dual-path changes output vs frozen-only ───────────────────────────
    noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl  = K_hat.unsqueeze(0)
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:2]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v_frozen = flow_model(noise_sp, t_ten, cond_gl)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                                   enhance_bias=None):
                v_dual = flow_model(noise_sp, t_ten, cond_gl)
            diff4 = (v_frozen.feats.float() - v_dual.feats.float()).abs()
            print(f'  [W4-DUAL  ] step t={t:.3f}: frozen vs dual-path '
                  f'max={diff4.max():.6f}  mean={diff4.mean():.6f}  '
                  f'{"paths differ ✓" if diff4.max() > 1e-8 else "IDENTICAL ← dual-path not firing"}')
            noise_sp = noise_sp.replace(noise_sp.feats - (t - t_prev) * v_frozen.feats)

    print('[WIRE CHECK] Done.\n')


# ── Gate 1 — backward compat ──────────────────────────────────────────────────

def gate1_backward_compat(flow_model, noise_sp, cond_gl, K_pooled,
                           lora_blocks, coords):
    """
    At alpha=0, B=zeros, enhance_bias=None:
      PATH A = q_base @ K_hat^T (identical to raw frozen)
      PATH B contributes 0
    => dual-path output == frozen output.  Tolerance 1e-2 accounts for fp16
    rounding between _attn_chunked (explicit float32) and xformers native.
    """
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

    x_base = _run(False)
    x_lora = _run(True)
    diff   = (x_base.feats.float() - x_lora.feats.float()).abs()
    tol    = 1e-2   # fp16 rounding noise from _attn_chunked vs xformers; real leak > 0.1
    print(f'  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}  tol={tol}')
    if diff.max() < tol:
        print('  [GATE 1] PASSED ✓  (LoRA plumbing clean at alpha=0)')
    else:
        raise RuntimeError(f'GATE 1 FAILED: max_diff={diff.max():.6f} — LoRA leaks at alpha=0')


# ── Gate 2 — gradient check ────────────────────────────────────────────────────

def gate2_grad_check(lora_blocks, alpha):
    print('\n[GATE 2] Gradient check...')
    print(f'  alpha.grad: '
          f'{"NONE ← FAIL" if alpha.grad is None else f"{alpha.grad.item():.6f}"}')

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

    if q_fail == 0 and kv_fail == 0 and alpha.grad is not None and frozen_nonzero == 0:
        print('  [GATE 2] PASSED ✓')
    else:
        print(f'  [GATE 2] issues: q_fail={q_fail} kv_fail={kv_fail} '
              f'frozen_nonzero={frozen_nonzero}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode',       type=str,   default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--rank',       type=int,   default=4)
    parser.add_argument('--loss_scale',   type=float, default=4096.0)
    parser.add_argument('--frame_stride', type=int,   default=1,
                        help='1 = all 150 frames')
    parser.add_argument('--smoke',        action='store_true',
                        help='1 epoch, frames [1,75,150] only, no checkpoint saved')
    args = parser.parse_args()

    mode         = args.mode
    EPOCHS       = args.epochs
    LR           = args.lr
    LORA_RANK    = args.rank
    LOSS_SCALE   = args.loss_scale
    FRAME_STRIDE = args.frame_stride
    SMOKE        = args.smoke

    print('=' * 72)
    print('Step 07b — MCFM LoRA Training (no enhancement)')
    print('=' * 72)
    print(f'  mode        : {mode}')
    print(f'  enhancement : NONE  (enhance_bias=None throughout)')
    print(f'  epochs      : {EPOCHS}')
    print(f'  lr          : {LR}')
    print(f'  lora_rank   : {LORA_RANK}')
    print(f'  loss_scale  : {LOSS_SCALE}  (fp16 FTZ fix)')
    print(f'  fixed_seed  : {FIXED_SEED}  (noise — same tensor every frame, every epoch)')
    print(f'  struct_seed : {STRUCT_SEED}')
    print(f'  results     : {_RESULTS}')
    print(f'  ckpt_dir    : {_CKPT_DIR}')
    print(f'  log         : {_RESULTS / _LOG_NAME}')

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('\n[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Voxel structure ───────────────────────────────────────────────────────
    print(f'\n[STRUCT] Sampling structure (STRUCT_SEED={STRUCT_SEED}, frame 75)...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}  coords_sum={coords.sum().item()}  (expect 7301, 661358)')
    assert N_vox == 7301, f'N_vox={N_vox} != 7301 — structure mismatch, check STRUCT_SEED'

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA setup ────────────────────────────────────────────────────────────
    print('\n[LORA] Building LoRA blocks...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, n_blocks=24).to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))

    gate0_verify(lora_blocks, alpha)

    n_trainable = count_trainable_v2(lora_blocks, alpha)
    n_frozen    = sum(p.numel() for p in flow_model.parameters())
    print(f'  trainable={n_trainable:,}  frozen={n_frozen:,}')

    # ── Optimizer + resume ────────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
    start_epoch  = 1
    loss_history = []
    latest_ckpt  = find_latest_ckpt()

    if latest_ckpt:
        print(f'\n[RESUME] Loading {latest_ckpt.name}...')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        # ── Resume compatibility guard ─────────────────────────────────────────
        if 'mode' in ckpt and ckpt['mode'] != mode:
            raise RuntimeError(
                f'RESUME MISMATCH: checkpoint mode={ckpt["mode"]} '
                f'but current --mode={mode}. Delete {_CKPT_DIR} or use correct --mode.')
        # guard: refuse to load checkpoints from the enhancement script (they have 'beta')
        if 'beta' in ckpt:
            raise RuntimeError(
                f'RESUME MISMATCH: checkpoint has beta={ckpt["beta"]} — '
                f'this is from the enhancement LoRA script (step07). '
                f'This script (step07b) trains without enhancement. '
                f'Delete {_CKPT_DIR} or point to results_mcfm_{{mode}}_noenh_lora_seed6/.')
        print(f'  [RESUME-CHECK] mode={mode} ✓  no-enhancement ✓')
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha     = nn.Parameter(ckpt['alpha'].to(DEVICE))
        optimizer = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        hist_path   = _RESULTS / 'loss_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                loss_history = json.load(f)
        print(f'  epoch={ckpt["epoch"]}  avg_loss={ckpt["avg_loss"]:.5f}  '
              f'alpha={alpha.item():.4f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs done. Nothing to train.')
        return

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] Building fixed noise (seed={FIXED_SEED})...')
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'  shape={tuple(fixed_noise_feats.shape)}  '
          f'mean={fixed_noise_feats.mean():.6f}  std={fixed_noise_feats.std():.6f}')
    print('  Same tensor cloned every frame every epoch (no stochasticity).')

    # ── DINOv2 encode all 150 frames ─────────────────────────────────────────
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    t_enc = time.time()
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            t = raw_tokens[i].float()
            print(f'  {i}/{N_FRAMES}  shape={tuple(t.shape)}  '
                  f'mean={t.mean():.5f}  std={t.std():.5f}')
    dino_model.cpu()
    torch.cuda.empty_cache()
    print(f'  Done in {time.time()-t_enc:.1f}s')
    t75 = raw_tokens[75].float()
    print(f'  [SPOT-CHECK frame 75] mean={t75.mean():.5f}  std={t75.std():.5f}  (expect ~0, ~1)')

    # ── K_pooled cache ────────────────────────────────────────────────────────
    print(f'\n[CACHE] Building K_pooled cache ({N_FRAMES} frames)...')
    kpooled_cache = {}
    for fi in range(1, N_FRAMES + 1):
        win = get_window(fi, mode)
        kpooled_cache[fi] = torch.stack([raw_tokens[i] for i in win], 0).mean(0).cpu()
        if fi in (1, 75, N_FRAMES):
            print(f'  frame {fi}: win={win}  K_pooled shape={tuple(kpooled_cache[fi].shape)}')
    print(f'  Done. {len(kpooled_cache)} entries.')

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = make_renderer()

    # ── Gate 1 — backward compat ──────────────────────────────────────────────
    print('\n[GATE 1] Checking LoRA plumbing...')
    K_hat_75, Kp_75, _ = blend(mode, raw_tokens, 75)
    noise_sp_g1 = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    gate1_backward_compat(flow_model, noise_sp_g1, K_hat_75.unsqueeze(0),
                          Kp_75, lora_blocks, coords)
    del noise_sp_g1, K_hat_75, Kp_75
    torch.cuda.empty_cache()

    # ── Wire check ────────────────────────────────────────────────────────────
    wire_check(mode, raw_tokens, flow_model,
               lora_blocks, alpha, coords, fixed_noise_feats)

    # ── Training loop ─────────────────────────────────────────────────────────
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.train()
    frame_list = [1, 75, 150] if SMOKE else list(range(1, N_FRAMES + 1, FRAME_STRIDE))

    print(f'\n[TRAIN] Epochs {start_epoch}→{EPOCHS}  '
          f'frames/epoch={len(frame_list)}  STEPS={STEPS}  '
          f'LOSS_SCALE={LOSS_SCALE}  FIXED_SEED={FIXED_SEED}  mode={mode}  '
          f'enhancement=NONE')

    gate2_done = False
    first_diag = True

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_loss = 0.0
        t_epoch    = time.time()

        for frame_i in frame_list:
            K_hat, K_pooled, win = blend(mode, raw_tokens, frame_i)
            cond_gl  = K_hat.unsqueeze(0)   # (1, 1374, 1024)

            noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)

            optimizer.zero_grad()

            # 24-step prefix, no grad
            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha
            )

            # Step 25 — val pass (no grad) for decode + GT loss
            x0_val   = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled,
                lora_blocks, alpha, require_grad=False
            )
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp

            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None: _flow_ref.cpu()
            gc.collect(); torch.cuda.empty_cache()

            try:
                color, slat_leaf_feats = decode_and_render(
                    pipeline, slat_val, renderer, diag=first_diag, device=DEVICE
                )
                gc.collect(); torch.cuda.empty_cache()

                gt   = load_gt_tensor(frame_i)
                loss = F.mse_loss(color, gt)

                if first_diag:
                    print(f'  [G4] loss={_gfn(loss)}  val={loss.item():.5f}')

                (loss * LOSS_SCALE).backward()
                if slat_leaf_feats.grad is not None:
                    slat_leaf_feats.grad.div_(LOSS_SCALE)

                if first_diag:
                    g = slat_leaf_feats.grad
                    print(f'  [G5] slat_leaf_feats.grad: '
                          f'{"None" if g is None else f"dtype={g.dtype} max={g.abs().max().item():.3e}"}')
                    first_diag = False

            finally:
                if _flow_ref is not None: _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render: no gradient on slat_leaf_feats')
            grad_slat_max = slat_leaf_feats.grad.abs().max().item()
            if grad_slat_max < 1e-30:
                print(f'  WARNING e{epoch:02d} f{frame_i:03d}: slat grad ZERO — LoRA will not update')

            # Step 25 — train pass (with grad for LoRA)
            grad_slat = slat_leaf_feats.grad.detach()
            x0_raw    = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled,
                lora_blocks, alpha, require_grad=True
            )
            slat = normalize_slat(x0_raw)
            torch.autograd.backward(slat.feats, grad_slat)
            torch.nn.utils.clip_grad_norm_(
                trainable_params_v2(lora_blocks, alpha), GRAD_CLIP
            )
            optimizer.step()

            if not gate2_done:
                gate2_grad_check(lora_blocks, alpha)
                gate2_done = True

            loss_val    = loss.item()
            epoch_loss += loss_val

            if frame_i == frame_list[0] or frame_i % 10 == 0:
                print(f'  e{epoch:02d} f{frame_i:03d}  loss={loss_val:.5f}  '
                      f'alpha={alpha.item():.4f}  win={win}')

            del color, slat_leaf_feats, gt, loss
            del x0_raw, slat, grad_slat
            del x_prefix, slat_val, cond_gl, K_pooled, K_hat
            gc.collect(); torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(frame_list)
        loss_history.append({'epoch': epoch, 'avg_loss': avg_loss,
                              'alpha': alpha.item()})
        elapsed = time.time() - t_epoch
        print(f'[EPOCH {epoch:02d}] avg_loss={avg_loss:.5f}  '
              f'alpha={alpha.item():.4f}  time={elapsed:.1f}s')

        # ── Save checkpoint ────────────────────────────────────────────────────
        if SMOKE:
            print('  [CKPT] smoke mode — skipping checkpoint save')
        else:
            ckpt_path = _CKPT_DIR / f'lora_e{epoch:03d}.pt'
            torch.save({
                'epoch'      : epoch,
                'mode'       : mode,
                # intentionally no 'beta' key — guards against mixing with step07 ckpts
                'lora_state' : lora_blocks.state_dict(),
                'alpha'      : alpha.data,
                'optimizer'  : optimizer.state_dict(),
                'avg_loss'   : avg_loss,
            }, ckpt_path)
            print(f'  [CKPT] saved {ckpt_path.name}')

        with open(_RESULTS / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    print(f'\n[DONE] Training complete. {EPOCHS} epochs. Results: {_RESULTS}')


if __name__ == '__main__':
    main()
