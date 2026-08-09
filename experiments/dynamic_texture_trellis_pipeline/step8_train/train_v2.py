"""
Step 8 v2 — LoRA training with fixed noise + soft enhancement bias.

BACKWARD COMPATIBLE: original train.py untouched. New checkpoints go to lora_ckpts_v2/.

Changes vs train.py:
  1. Fixed noise seed=6, same tensor every frame every epoch
  2. Soft additive enhance bias (beta frozen from step06 sweep), PATH A only
  3. Uses lora_v2 + dual_path_v2 (delta-only LoRA, clean separation)
  4. Alignment matrix from TRAIN_CONFIG (must match smoothed-token config)
  5. Gates 0-4 with full logging

SET BEFORE RUNNING:
  BETA_WINNER  — winner from step06b 150-frame sweep (e.g. 4.0)
  TRAIN_CONFIG — winning phase C config (e.g. 'C3'), determines alignment matrix

Run:
  cd .../step8_train
  SPCONV_ALGO=native ATTN_BACKEND=xformers python train_v2.py
"""

# ── Tee BEFORE all imports ────────────────────────────────────────────────────
import sys, os
from pathlib import Path

os.environ['SPCONV_ALGO']            = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']               = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']        = '1'
os.environ['TRANSFORMERS_OFFLINE']  = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE     = Path(__file__).resolve().parent
_PIPE     = _HERE.parent
_ROOT     = _PIPE.parent.parent
RESULTS_V2 = _HERE.parent / 'results_v2'
RESULTS_V2.mkdir(exist_ok=True)

# ── SET THESE BEFORE RUNNING ──────────────────────────────────────────────────
BETA_WINNER  = None   # e.g. 4.0  — winner from step06b sweep
TRAIN_CONFIG = None   # e.g. 'C3' — winning phase C config (determines alignment matrix)
# ─────────────────────────────────────────────────────────────────────────────

_LOG_NAME = (f'train_v2'
             f'_cfg{TRAIN_CONFIG or "UNSET"}'
             f'_beta{BETA_WINNER or "UNSET"}'
             f'_seed6.log')


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

_tee = _Tee(RESULTS_V2 / _LOG_NAME)
sys.stdout = _tee
sys.stderr = _tee

# ── imports after Tee ─────────────────────────────────────────────────────────
import json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

from step1_input_prep.input_prep       import (N_FRAMES, load_frame,
                                                t_to_frame_idx, get_window_indices)
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step6_5_lora.lora_v2               import (build_lora_blocks, freeze_trellis,
                                                 gate0_verify, trainable_params_v2,
                                                 count_trainable_v2)
from step6_5_lora.dual_path_v2          import dual_path_ctx_v2, build_soft_enhance_bias
from step8_decode_render.decode_render  import (make_renderer, normalize_slat,
                                                decode_and_render, _gfn,
                                                RENDER_RES, EXTRINSICS, INTRINSICS,
                                                SLAT_MEAN, SLAT_STD)

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
ARTIFACTS_DIR = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments'
                     '/enhancement/artifacts')

DEVICE       = torch.device('cuda')
LORA_RANK    = 4
LR           = 1e-4
EPOCHS       = 50
FIXED_SEED   = 6          # same noise for ALL frames, ALL epochs
STRUCT_SEED  = 42         # voxel structure — matches _base.py
GRAD_CLIP    = 1.0
LOSS_SCALE   = 4096.0     # fp16 FTZ fix — do NOT remove
K            = 3          # MCFM window half-width
FRAME_STRIDE = 1
STEPS        = 25
RESCALE_T    = 3.0
CKPT_DIR     = RESULTS_V2 / 'lora_ckpts_v2'
CKPT_DIR.mkdir(exist_ok=True)

# RENDER_RES, EXTRINSICS, INTRINSICS, SLAT_MEAN, SLAT_STD — imported from step8_decode_render

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]


# ── Alignment matrix path ─────────────────────────────────────────────────────

def resolve_alignment_path(config: str) -> Path:
    """
    Returns path to the alignment matrix for the given training config.
    config: 'raw', 'C0'..'C10', 'C_v2', 'D0'..'D6', 'D_v2'
    """
    if config == 'raw':
        return ARTIFACTS_DIR / 'voxel_to_token.pt'
    if config.startswith('C') and config != 'C_v2':
        return ARTIFACTS_DIR / 'phase_c' / 'v1' / config / 'voxel_to_token.pt'
    if config == 'C_v2':
        return ARTIFACTS_DIR / 'phase_c' / 'v2' / 'voxel_to_token.pt'
    if config.startswith('D') and config != 'D_v2':
        return ARTIFACTS_DIR / 'phase_d' / 'v1' / config / 'voxel_to_token.pt'
    if config == 'D_v2':
        return ARTIFACTS_DIR / 'phase_d' / 'v2' / 'voxel_to_token.pt'
    raise ValueError(f'Unknown config: {config}')


# ── Helpers ───────────────────────────────────────────────────────────────────
# normalize_slat, decode_and_render, _gfn  ← imported from step8_decode_render

def load_gt_frame(frame_idx: int) -> torch.Tensor:
    img = load_frame(frame_idx).resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled,
                           lora_blocks, alpha, enhance_bias):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                                   enhance_bias=enhance_bias):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled,
                      lora_blocks, alpha, enhance_bias, require_grad):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx = dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                            enhance_bias=enhance_bias)
    if require_grad:
        with ctx:
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with ctx:
                v = flow_model(x_in, t_ten, cond_gl)
    x0_feats = x_in.feats - (t - t_prev) * v.feats
    return x_in.replace(x0_feats)


def find_latest_ckpt():
    ckpts = sorted(CKPT_DIR.glob('lora_v2_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── GATE 1 — backward compat check ───────────────────────────────────────────

def gate1_backward_compat(flow_model, noise_sp, cond_gl, K_pooled,
                           lora_blocks, enhance_bias, coords, N_vox):
    """
    Verify: alpha=0 + B=zeros → output identical to no-LoRA baseline within fp16 tol.
    Runs ONE full 25-step denoise, compares rendered pixel values.
    """
    print('\n[GATE 1] Backward compat check (alpha=0, B=zeros)...')

    alpha_zero = nn.Parameter(torch.tensor(0.0, device=DEVICE))

    def _denoise_full(use_lora):
        x = sp.SparseTensor(feats=noise_sp.feats.clone(), coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                if use_lora:
                    with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks,
                                          alpha_zero, enhance_bias=enhance_bias):
                        v = flow_model(x, t_ten, cond_gl)
                else:
                    v = flow_model(x, t_ten, cond_gl)
                x = x.replace(x.feats - (t - t_prev) * v.feats)
        return x

    x0_baseline = _denoise_full(use_lora=False)
    x0_lora     = _denoise_full(use_lora=True)

    diff = (x0_baseline.feats.float() - x0_lora.feats.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f'  [GATE 1] slat max_abs_diff  : {max_diff:.6f}')
    print(f'  [GATE 1] slat mean_abs_diff : {mean_diff:.6f}')

    tol = 1e-3
    if max_diff < tol:
        print(f'  [GATE 1] PASSED  (max_diff={max_diff:.2e} < tol={tol})  ✓\n')
    else:
        print(f'  [GATE 1] FAILED  (max_diff={max_diff:.2e} >= tol={tol})  ← plumbing leak!\n')
        raise RuntimeError(f'GATE 1 FAILED: max_abs_diff={max_diff:.6f} — LoRA plumbing leaks at alpha=0')

    return max_diff


# ── GATE 2 — first backward grad check ───────────────────────────────────────

def gate2_grad_check(lora_blocks, alpha):
    print('\n[GATE 2] First-backward gradient check...')

    # alpha grad
    if alpha.grad is None:
        print('  [GATE 2] alpha.grad : NONE  ← FAIL')
    else:
        print(f'  [GATE 2] alpha.grad : {alpha.grad.item():.4f}  (historical ~135)')

    # lora_q and lora_kv A/B grads
    q_fails = 0; kv_fails = 0
    q_A_vals = []; q_B_vals = []; kv_A_vals = []; kv_B_vals = []
    for i, blk in enumerate(lora_blocks):
        for name, p in [('lora_q.A', blk.lora_q.A), ('lora_q.B', blk.lora_q.B),
                        ('lora_kv.A', blk.lora_kv.A), ('lora_kv.B', blk.lora_kv.B)]:
            if p.grad is None or p.grad.abs().max().item() == 0:
                print(f'  [GATE 2] block {i:02d} {name} ZERO GRAD  ← FAIL')
                if 'q.' in name: q_fails += 1
                else: kv_fails += 1
            mx = p.grad.abs().max().item() if p.grad is not None else 0.0
            if 'q.A' in name:   q_A_vals.append(mx)
            elif 'q.B' in name: q_B_vals.append(mx)
            elif 'kv.A' in name: kv_A_vals.append(mx)
            elif 'kv.B' in name: kv_B_vals.append(mx)

    def _stat(vals, name):
        if not vals: return
        print(f'  [GATE 2] {name:20s}: min={min(vals):.3e}  max={max(vals):.3e}  '
              f'mean={sum(vals)/len(vals):.3e}')

    _stat(q_A_vals,  'lora_q  A grad')
    _stat(q_B_vals,  'lora_q  B grad')
    _stat(kv_A_vals, 'lora_kv A grad')
    _stat(kv_B_vals, 'lora_kv B grad')

    # frozen params
    frozen_with_grad = sum(
        1 for n, p in lora_blocks.named_parameters()
        if not p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    )
    print(f'  [GATE 2] frozen params with nonzero grad : {frozen_with_grad}  (expected 0)')

    if q_fails == 0 and kv_fails == 0 and alpha.grad is not None and frozen_with_grad == 0:
        print('  [GATE 2] PASSED  ✓\n')
    else:
        print(f'  [GATE 2] ISSUES: q_fails={q_fails} kv_fails={kv_fails}  '
              f'frozen_nonzero={frozen_with_grad}\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Startup checks ────────────────────────────────────────────────────────
    print('=' * 72)
    print('Step 8 v2 — Fixed-Noise LoRA Training')
    print('=' * 72)

    if BETA_WINNER is None:
        raise ValueError('BETA_WINNER not set — run step06b sweep first and set at top of file')
    if TRAIN_CONFIG is None:
        raise ValueError('TRAIN_CONFIG not set — run visual inspection first and set at top of file')

    align_path = resolve_alignment_path(TRAIN_CONFIG)
    if not align_path.exists():
        raise FileNotFoundError(f'Alignment matrix not found: {align_path}')

    print(f'  BETA_WINNER  : {BETA_WINNER}')
    print(f'  TRAIN_CONFIG : {TRAIN_CONFIG}')
    print(f'  align_path   : {align_path}')
    print(f'  FIXED_SEED   : {FIXED_SEED}  (same noise all frames all epochs)')
    print(f'  LOSS_SCALE   : {LOSS_SCALE}')
    print(f'  EPOCHS       : {EPOCHS}')
    print(f'  CKPT_DIR     : {CKPT_DIR}')
    print(f'  LOG          : {RESULTS_V2 / _LOG_NAME}')

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('\n[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Voxel structure ───────────────────────────────────────────────────────
    print(f'\n[LOAD] Sampling voxel structure (STRUCT_SEED={STRUCT_SEED}, frame 75)...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}')

    # Free unused pipeline models
    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print('\n[LORA] Building LoRA blocks...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, n_blocks=24).to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))

    # GATE 0
    gate0_verify(lora_blocks, alpha)

    total_trainable = count_trainable_v2(lora_blocks, alpha)
    print(f'[LORA] trainable={total_trainable:,}  frozen={sum(p.numel() for p in flow_model.parameters()):,}')

    # ── Optimizer + resume ────────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
    start_epoch  = 1
    loss_history = []
    latest_ckpt  = find_latest_ckpt()
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha     = nn.Parameter(ckpt['alpha'].to(DEVICE))
        optimizer = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=LR)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        log_path = RESULTS_V2 / 'loss_history_v2.json'
        if log_path.exists():
            with open(log_path) as f:
                loss_history = json.load(f)
        print(f'[RESUME] {latest_ckpt.name}  epoch={ckpt["epoch"]}  '
              f'avg_loss={ckpt["avg_loss"]:.5f}  alpha={alpha.item():.4f}')
    else:
        print('[RESUME] No checkpoint found — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs complete.')
        return

    # ── Alignment matrix + enhancement bias ──────────────────────────────────
    print(f'\n[ALIGN] Loading {align_path}...')
    vox_to_tok = torch.load(align_path, weights_only=True).to(DEVICE)
    print(f'  vox_to_tok shape={tuple(vox_to_tok.shape)}  '
          f'range=[{int(vox_to_tok.min())}, {int(vox_to_tok.max())}]  '
          f'unique={vox_to_tok.unique().numel()}')

    enhance_bias = build_soft_enhance_bias(vox_to_tok, BETA_WINNER, DEVICE)
    print(f'  enhance_bias shape={tuple(enhance_bias.shape)}  beta={BETA_WINNER}  (PATH A only)')

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] Building fixed noise tensor (seed={FIXED_SEED})...')
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'  shape={tuple(fixed_noise_feats.shape)}  '
          f'mean={fixed_noise_feats.mean():.6f}  std={fixed_noise_feats.std():.6f}  '
          f'first5={fixed_noise_feats[0, :5].tolist()}')
    print(f'  This EXACT tensor is used for ALL frames and ALL epochs.')

    # ── DINOv2 + pre-encode ───────────────────────────────────────────────────
    print(f'\n[ENCODE] Pre-encoding {N_FRAMES} frames...')
    dino = load_dino(DEVICE)   # same as train.py — always load fresh from local cache

    t0 = time.time()
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    print(f'  Done in {time.time()-t0:.1f}s')
    del dino
    torch.cuda.empty_cache()

    # ── Projector (frozen) ────────────────────────────────────────────────────
    projector = FrameWeightProjector(k=K).to(DEVICE)
    for p in projector.parameters():
        p.requires_grad_(False)

    # ── K_pooled cache (150 entries, WITH repeats per spec) ───────────────────
    print('\n[CACHE] Building K_pooled cache (150 frames, WITH boundary repeats)...')
    kpooled_cache = {}
    for frame_i in range(1, N_FRAMES + 1):
        win_idx = get_window_indices(frame_i, K)
        toks_gpu = {idx: {k: v.to(DEVICE) for k, v in all_tokens[idx].items()}
                    for idx in win_idx}
        # WITH repeats — dict.fromkeys REMOVED, matches MCFM boundary weighting
        kpooled_cache[frame_i] = torch.stack(
            [toks_gpu[i]['tokens'] for i in win_idx]
        ).mean(0).cpu()
        if frame_i == 1:
            print(f'  frame 1 win_idx={win_idx}  (repeats at boundary: intentional)')
        if frame_i % 50 == 0:
            print(f'  {frame_i}/{N_FRAMES}')
    torch.cuda.empty_cache()
    print(f'  K_pooled cache: {len(kpooled_cache)} entries, each (1374, 1024)')

    # ── Renderer (step8_decode_render) ────────────────────────────────────────
    renderer = make_renderer()

    # ── GATE 1 — backward compat ──────────────────────────────────────────────
    noise_sp_gate = sp.SparseTensor(
        feats=fixed_noise_feats.clone(), coords=coords
    )
    tok75     = {75: {k: v.to(DEVICE) for k, v in all_tokens[75].items()}}
    K_hat_75, _ = run_mcfm('v2', tok75, [75], 75, torch.ones(1, device=DEVICE))
    cond_75   = K_hat_75.unsqueeze(0)
    Kp_75     = kpooled_cache[75].to(DEVICE)

    gate1_backward_compat(flow_model, noise_sp_gate, cond_75, Kp_75,
                          lora_blocks, enhance_bias, coords, N_vox)

    # ── Training ──────────────────────────────────────────────────────────────
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.train()
    frame_list = list(range(1, N_FRAMES + 1, FRAME_STRIDE))

    print(f'\n[TRAIN] Starting epoch {start_epoch} → {EPOCHS}  '
          f'({len(frame_list)} frames/epoch  STEPS={STEPS}  '
          f'LOSS_SCALE={LOSS_SCALE}  FIXED_SEED={FIXED_SEED})')

    gate2_done  = False
    first_diag  = True   # G4/G5 loss+grad prints fire once then stop

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_loss = 0.0
        t_epoch    = time.time()

        for frame_i in frame_list:
            t_val     = (frame_i - 1) / (N_FRAMES - 1)
            frame_idx = t_to_frame_idx(t_val)
            win_idx   = get_window_indices(frame_idx, K)

            tokens_gpu = {idx: {k: v.to(DEVICE) for k, v in all_tokens[idx].items()}
                          for idx in win_idx}
            lambda_vec, _ = get_frame_weights(t_val, frame_idx, win_idx,
                                              tokens_gpu, projector)
            K_hat, _ = run_mcfm('v2', tokens_gpu, win_idx, frame_idx, lambda_vec)
            cond_gl  = K_hat.unsqueeze(0)

            K_pooled = kpooled_cache[frame_idx].to(DEVICE)

            # Fixed noise — clone so denoising loop doesn't alias
            noise_sp = sp.SparseTensor(
                feats=fixed_noise_feats.clone(), coords=coords
            )

            # ── Forward (kept identical to train.py pattern) ──────────────
            optimizer.zero_grad()

            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled,
                lora_blocks, alpha, enhance_bias
            )
            x0_val = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled,
                lora_blocks, alpha, enhance_bias, require_grad=False
            )
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp

            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None:
                _flow_ref.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # diag=True → step8_decode_render prints S8|G0..M3 once, then auto-off
                color, slat_leaf_feats = decode_and_render(
                    pipeline, slat_val, renderer, diag=first_diag, device=DEVICE
                )
                gc.collect()
                torch.cuda.empty_cache()

                gt   = load_gt_frame(frame_idx)
                loss = F.mse_loss(color, gt)

                if first_diag:
                    print(f'  [G4] loss: {_gfn(loss)}  val={loss.item():.5f}')

                (loss * LOSS_SCALE).backward()
                if slat_leaf_feats.grad is not None:
                    slat_leaf_feats.grad.div_(LOSS_SCALE)

                if first_diag:
                    g = slat_leaf_feats.grad
                    print(f'  [G5] slat_leaf_feats.grad: '
                          f'{"None" if g is None else f"dtype={g.dtype} max={g.abs().max().item():.3e}"}')
                    first_diag = False

            finally:
                if _flow_ref is not None:
                    _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render: no gradient on slat_leaf_feats')
            grad_slat_max = slat_leaf_feats.grad.abs().max().item()
            if grad_slat_max < 1e-30:
                print(f'  WARNING e{epoch:02d} f{frame_i:03d}: slat grad ALL ZERO — LoRA will not update')

            grad_slat = slat_leaf_feats.grad.detach()
            x0_raw    = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled,
                lora_blocks, alpha, enhance_bias, require_grad=True
            )
            slat = normalize_slat(x0_raw)
            torch.autograd.backward(slat.feats, grad_slat)
            torch.nn.utils.clip_grad_norm_(
                trainable_params_v2(lora_blocks, alpha), GRAD_CLIP
            )
            optimizer.step()

            # GATE 2 — first backward only
            if not gate2_done:
                gate2_grad_check(lora_blocks, alpha)
                gate2_done = True

            loss_val    = loss.item()
            epoch_loss += loss_val
            loss_history.append({
                'epoch': epoch, 'frame': frame_i,
                'loss': loss_val, 'alpha': alpha.item(),
            })

            if frame_i == frame_list[0] or frame_i % 10 == 0:
                print(f'  e{epoch:02d} f{frame_i:03d}  loss={loss_val:.5f}'
                      f'  alpha={alpha.item():.4f}')

            del color, slat_leaf_feats, gt, loss
            del x0_raw, slat, grad_slat
            del x_prefix, slat_val, cond_gl, K_pooled
            del tokens_gpu, lambda_vec
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(len(frame_list), 1)
        elapsed  = time.time() - t_epoch
        print(f'Epoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.5f}'
              f'  alpha={alpha.item():.4f}  t={elapsed:.0f}s')

        ckpt_path = CKPT_DIR / f'lora_v2_e{epoch:03d}.pt'
        torch.save({
            'epoch':       epoch,
            'avg_loss':    avg_loss,
            'alpha':       alpha.detach().cpu(),
            'lora_state':  lora_blocks.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'beta_winner': BETA_WINNER,
            'train_config': TRAIN_CONFIG,
            'fixed_seed':  FIXED_SEED,
        }, ckpt_path)
        print(f'  Saved: {ckpt_path}')

        with open(RESULTS_V2 / 'loss_history_v2.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    print(f'\n[DONE] Final alpha: {alpha.item():.4f}')
    print(f'[DONE] Results: {RESULTS_V2}')


if __name__ == '__main__':
    main()
