"""
Single-frame LoRA sanity check — no token blending, no enhance_bias.

Purpose:
  Verify end-to-end gradient flow: LoRA params actually learn from one frame.
  Raw DINOv2 tokens → flow model (LoRA) → SLaT → mesh → render → MSE vs GT.
  If loss goes down, the full pipeline is wired correctly.

BACKWARD COMPATIBLE: no existing files touched.
Checkpoints go to results_single_frame/lora_sf_frame{F}/.

Run:
  cd .../step8_train
  SPCONV_ALGO=native ATTN_BACKEND=xformers python train_single_frame.py --frame 75

Submit:
  sbatch submit_single_frame.sh
"""

# ── Tee BEFORE all imports ────────────────────────────────────────────────────
import sys, os, argparse as _ap
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent
_ROOT = _PIPE.parent.parent

# ── Parse args early (needed for log path) ───────────────────────────────────
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--frame',  type=int,   default=75)
_pre.add_argument('--epochs', type=int,   default=30)
_pre.add_argument('--lr',     type=float, default=1e-4)
_pre.add_argument('--seed',   type=int,   default=6)
_pre.add_argument('--rank',   type=int,   default=4)
_ARGS, _ = _pre.parse_known_args()

_RESULTS = _HERE.parent / 'results_single_frame'
_RESULTS.mkdir(exist_ok=True)
_CKPT_DIR = _RESULTS / f'lora_sf_frame{_ARGS.frame:04d}'
_CKPT_DIR.mkdir(exist_ok=True)
_LOG_PATH = _RESULTS / f'train_sf_frame{_ARGS.frame:04d}_e{_ARGS.epochs}_lr{_ARGS.lr}_seed{_ARGS.seed}.log'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

sys.stdout = _Tee(_LOG_PATH)
sys.stderr = sys.stdout

# ── Imports ───────────────────────────────────────────────────────────────────
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

from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step1_input_prep.input_prep       import load_frame
from step6_5_lora.lora_v2              import (build_lora_blocks, freeze_trellis,
                                                gate0_verify, trainable_params_v2,
                                                count_trainable_v2)
from step6_5_lora.dual_path_v2         import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render, _gfn,
                                               RENDER_RES, SLAT_MEAN, SLAT_STD)

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
STRUCT_SEED = 42
LOSS_SCALE  = 4096.0
STEPS       = 25
RESCALE_T   = 3.0
GRAD_CLIP   = 1.0

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]


# ── Denoising ─────────────────────────────────────────────────────────────────

def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled, lora_blocks, alpha, require_grad):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx = dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha)
    if require_grad:
        with ctx:
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with ctx:
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


# ── GATE 2 — grad check after first backward ──────────────────────────────────

def gate2_grad_check(lora_blocks, alpha):
    print('\n[GATE 2] First-backward gradient check...')

    if alpha.grad is None:
        print('  [GATE 2] alpha.grad : NONE  ← FAIL')
    else:
        print(f'  [GATE 2] alpha.grad : {alpha.grad.item():.6f}')

    total = 0; zeros = 0
    q_A_mx = []; q_B_mx = []; kv_A_mx = []; kv_B_mx = []
    for blk in lora_blocks:
        for name, p in [('lora_q.A',  blk.lora_q.A),  ('lora_q.B',  blk.lora_q.B),
                         ('lora_kv.A', blk.lora_kv.A), ('lora_kv.B', blk.lora_kv.B)]:
            total += 1
            mx = p.grad.abs().max().item() if p.grad is not None else 0.0
            if mx == 0.0: zeros += 1
            if   'q.A'  in name: q_A_mx.append(mx)
            elif 'q.B'  in name: q_B_mx.append(mx)
            elif 'kv.A' in name: kv_A_mx.append(mx)
            elif 'kv.B' in name: kv_B_mx.append(mx)

    def _s(vals, label):
        if vals:
            print(f'  [GATE 2] {label:20s}: min={min(vals):.3e}  max={max(vals):.3e}  '
                  f'mean={sum(vals)/len(vals):.3e}')
    _s(q_A_mx,  'lora_q  A grad')
    _s(q_B_mx,  'lora_q  B grad')
    _s(kv_A_mx, 'lora_kv A grad')
    _s(kv_B_mx, 'lora_kv B grad')

    frozen_nonzero = sum(
        1 for p in lora_blocks.parameters()
        if not p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    )
    print(f'  [GATE 2] zero-grad params   : {zeros}/{total}  (expected 0)')
    print(f'  [GATE 2] frozen nonzero grad: {frozen_nonzero}  (expected 0)')

    if zeros == 0 and alpha.grad is not None and frozen_nonzero == 0:
        print('  [GATE 2] PASSED  ✓\n')
    else:
        print('  [GATE 2] ISSUES — check wiring\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--frame',  type=int,   default=75,   help='GT frame index [1..150]')
    parser.add_argument('--epochs', type=int,   default=30,   help='training epochs')
    parser.add_argument('--lr',     type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--seed',   type=int,   default=6,    help='fixed noise seed')
    parser.add_argument('--rank',   type=int,   default=4,    help='LoRA rank')
    args = parser.parse_args()

    print('=' * 72)
    print('Single-Frame LoRA Sanity Check — no blending, no enhance_bias')
    print('=' * 72)
    print(f'  frame      : {args.frame}')
    print(f'  epochs     : {args.epochs}')
    print(f'  lr         : {args.lr}')
    print(f'  noise seed : {args.seed}  (fixed — same every epoch)')
    print(f'  lora rank  : {args.rank}')
    print(f'  LOSS_SCALE : {LOSS_SCALE}')
    print(f'  CKPT_DIR   : {_CKPT_DIR}')
    print(f'  LOG        : {_LOG_PATH}')

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('\n[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    print(f'  flow_model.in_channels = {flow_model.in_channels}')

    # ── Voxel structure (same seed as all other scripts) ─────────────────────
    print(f'\n[STRUCT] Sampling voxel structure (STRUCT_SEED={STRUCT_SEED}, frame {args.frame})...')
    img_struct  = load_frame(args.frame)
    cond_struct = pipeline.get_cond([img_struct])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}')

    # Free models not needed for training
    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print(f'\n[LORA] Building LoRA blocks (rank={args.rank})...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=args.rank, n_blocks=24).to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))
    gate0_verify(lora_blocks, alpha)
    print(f'  trainable = {count_trainable_v2(lora_blocks, alpha):,}')
    print(f'  frozen    = {sum(p.numel() for p in flow_model.parameters()):,}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=args.lr)
    latest_ckpt = sorted(_CKPT_DIR.glob('lora_sf_e*.pt'))
    start_epoch = 1
    loss_history = []
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt[-1], map_location=DEVICE)
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha       = nn.Parameter(ckpt['alpha'].to(DEVICE))
        optimizer   = torch.optim.Adam(trainable_params_v2(lora_blocks, alpha), lr=args.lr)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f'[RESUME] {latest_ckpt[-1].name}  epoch={ckpt["epoch"]}  '
              f'loss={ckpt["loss"]:.6f}  alpha={alpha.item():.4f}')
    else:
        print('[RESUME] No checkpoint — starting fresh.')

    if start_epoch > args.epochs:
        print(f'All {args.epochs} epochs complete.')
        return

    # ── DINOv2 encode target frame ────────────────────────────────────────────
    print(f'\n[DINO] Encoding frame {args.frame}...')
    dino = load_dino(DEVICE)
    gt_pil  = load_frame(args.frame)
    tokens  = encode_frames([args.frame], [gt_pil], dino, DEVICE)
    del dino
    torch.cuda.empty_cache()

    tok = tokens[args.frame]
    print(f'  tokens shape : {tuple(tok["tokens"].shape)}')
    print(f'  cls    shape : {tuple(tok["cls"].shape)}')

    # cond_gl  : (1, 1374, 1024) — raw single-frame tokens, no blending
    # K_pooled : (1374, 1024)    — same tokens for PATH B (no window mean)
    cond_gl  = tok['tokens'].unsqueeze(0)     # (1, 1374, 1024)
    K_pooled = tok['tokens']                  # (1374, 1024)
    print(f'  cond_gl  shape: {tuple(cond_gl.shape)}  (raw, no blending)')
    print(f'  K_pooled shape: {tuple(K_pooled.shape)}  (= cond_gl squeezed, no averaging)')

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] Building fixed noise (seed={args.seed})...')
    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'  shape={tuple(fixed_noise_feats.shape)}'
          f'  mean={fixed_noise_feats.mean():.4f}'
          f'  std={fixed_noise_feats.std():.4f}')
    print(f'  Same tensor reused every epoch — true single-frame overfit test.')

    # ── GT image ──────────────────────────────────────────────────────────────
    gt_img = gt_pil.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt_tensor = (torch.from_numpy(np.array(gt_img)).float()
                 .div(255.0).permute(2, 0, 1).to(DEVICE))   # (3, H, W)
    print(f'\n[GT] Frame {args.frame}: {tuple(gt_tensor.shape)}'
          f'  range=[{gt_tensor.min():.3f}, {gt_tensor.max():.3f}]')

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = make_renderer()

    # ── Training ──────────────────────────────────────────────────────────────
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.train()
    gate2_done = False
    first_diag = True

    print(f'\n[TRAIN] Epochs {start_epoch} → {args.epochs}')
    t0_train = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        optimizer.zero_grad()

        # ── 24 prefix steps no_grad ───────────────────────────────────────────
        noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        x_prefix = denoise_prefix_nograd(
            flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha
        )

        # ── Last step no_grad → decode → render → loss ────────────────────────
        x0_val   = denoise_last_step(
            flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha, require_grad=False
        )
        slat_val = normalize_slat(x0_val)
        del x0_val, noise_sp

        _flow_ref = pipeline.models.get('slat_flow_model')
        if _flow_ref is not None:
            _flow_ref.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            color, slat_leaf_feats = decode_and_render(
                pipeline, slat_val, renderer, diag=first_diag, device=DEVICE
            )
            gc.collect()
            torch.cuda.empty_cache()

            loss = F.mse_loss(color, gt_tensor)

            if first_diag:
                print(f'  [G4] loss tensor: {_gfn(loss)}  val={loss.item():.6f}')

            (loss * LOSS_SCALE).backward()
            if slat_leaf_feats.grad is not None:
                slat_leaf_feats.grad.div_(LOSS_SCALE)

            if first_diag:
                g = slat_leaf_feats.grad
                print(f'  [G5] slat_leaf_feats.grad: '
                      f'{"None ← FAIL" if g is None else f"max={g.abs().max().item():.3e}  mean={g.abs().mean().item():.3e}"}')
                first_diag = False

        finally:
            if _flow_ref is not None:
                _flow_ref.to(DEVICE)

        if slat_leaf_feats.grad is None:
            raise RuntimeError('slat_leaf_feats.grad is None — gradient did not flow')

        grad_slat_max = slat_leaf_feats.grad.abs().max().item()
        if grad_slat_max < 1e-30:
            print(f'  WARNING e{epoch:03d}: slat grad ALL ZERO — LoRA will not update')

        grad_slat = slat_leaf_feats.grad.detach()

        # ── Last step WITH grad → inject ──────────────────────────────────────
        x0_raw = denoise_last_step(
            flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha, require_grad=True
        )
        slat = normalize_slat(x0_raw)
        torch.autograd.backward(slat.feats, grad_slat)
        torch.nn.utils.clip_grad_norm_(trainable_params_v2(lora_blocks, alpha), GRAD_CLIP)
        optimizer.step()

        # GATE 2 — first backward only
        if not gate2_done:
            gate2_grad_check(lora_blocks, alpha)
            gate2_done = True

        loss_val = loss.item()
        elapsed  = time.time() - t_epoch
        loss_history.append({'epoch': epoch, 'loss': loss_val, 'alpha': alpha.item()})
        print(f'  epoch {epoch:03d}/{args.epochs}  loss={loss_val:.6f}'
              f'  alpha={alpha.item():.4f}  t={elapsed:.1f}s')

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_path = _CKPT_DIR / f'lora_sf_e{epoch:03d}.pt'
        torch.save({
            'epoch':        epoch,
            'loss':         loss_val,
            'alpha':        alpha.detach().cpu(),
            'lora_state':   lora_blocks.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'frame':        args.frame,
            'seed':         args.seed,
            'loss_history': loss_history,
        }, ckpt_path)

        del color, slat_leaf_feats, loss, x0_raw, slat, grad_slat
        del x_prefix, slat_val
        gc.collect()
        torch.cuda.empty_cache()

    total_min = (time.time() - t0_train) / 60
    print(f'\n[DONE] {args.epochs} epochs in {total_min:.1f} min')
    print(f'[DONE] Final loss  : {loss_history[-1]["loss"]:.6f}')
    print(f'[DONE] First loss  : {loss_history[0]["loss"]:.6f}')
    ratio = loss_history[-1]["loss"] / max(loss_history[0]["loss"], 1e-9)
    print(f'[DONE] Loss ratio  : {ratio:.4f}  (< 1.0 = learning)')
    print(f'[DONE] Final alpha : {alpha.item():.4f}')
    print(f'[DONE] Checkpoints : {_CKPT_DIR}')

    with open(_RESULTS / f'loss_history_frame{args.frame:04d}.json', 'w') as f:
        json.dump(loss_history, f, indent=2)


if __name__ == '__main__':
    main()
