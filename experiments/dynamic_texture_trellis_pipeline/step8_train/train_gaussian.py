"""
Step 8 (Gaussian): Training loop using Gaussian decoder + GaussianRenderer.

Gradient path:
  loss ← MSE ← rendered_color ← GaussianRenderer ← Gaussian params (fp32)
       ← SLatGaussianDecoder ← SLaT_x0 (last step, grad) ← LoRA params + alpha

Checkpoints saved to lora_ckpts_gs/ (separate from mesh-trained lora_ckpts/).

Run from this directory:
  cd .../step8_train
  SPCONV_ALGO=native ATTN_BACKEND=xformers python train_gaussian.py 2>&1 | tee ../results/train_step8_gs.log
"""

import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

os.environ['SPCONV_ALGO']               = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']                  = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']           = '1'
os.environ['TRANSFORMERS_OFFLINE']     = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.renderers import GaussianRenderer
import utils3d.torch as u3d

from step1_input_prep.input_prep       import (prepare_input, N_FRAMES, load_frame,
                                                t_to_frame_idx, get_window_indices)
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from step6_attn_enhancement.enhancement import build_enhancement_matrix
from step6_5_lora.lora                  import insert_lora, trainable_params, count_trainable
from step6_5_lora.ray_attention         import dual_path_ctx

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED   = 'microsoft/TRELLIS-image-large'
GT_FRAME_75  = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                '/outputs/teapot_lava_kling_premium'
                '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
RESULTS_DIR  = Path('../results')
CKPT_DIR     = RESULTS_DIR / 'lora_ckpts_gs'   # separate from mesh-trained checkpoints
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

K            = 3
LORA_RANK    = 4
LR           = 1e-4
EPOCHS       = 50
NOISE_SEED   = 42
RENDER_RES   = 518
GRAD_CLIP    = 1.0
DEVICE       = torch.device('cuda')

# Keep LOSS_SCALE as a safety net; gs_decoder runs in fp32 so underflow is
# unlikely, but scale is harmless and keeps the numeric regime identical to train.py.
LOSS_SCALE   = 4096.0

STEPS        = 25
RESCALE_T    = 3.0
FRAME_STRIDE = 1       # all 150 frames per epoch

SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,  1.2039363384246826
], dtype=torch.float32)
SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338
], dtype=torch.float32)

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt_frame(frame_idx: int) -> torch.Tensor:
    img = load_frame(frame_idx).resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled, lora_blocks, alpha, E, require_grad):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    if require_grad:
        with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
                v = flow_model(x_in, t_ten, cond_gl)
    x0_feats = x_in.feats - (t - t_prev) * v.feats
    return x_in.replace(x0_feats)


def normalize_slat(x0: sp.SparseTensor) -> sp.SparseTensor:
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


_DIAG_DONE  = False
_DIAG_HOOKS = []

def _gfn(t):
    if t is None: return 'None'
    return (f'req={t.requires_grad} '
            f'fn={type(t.grad_fn).__name__ if t.grad_fn else "None"} '
            f'shape={tuple(t.shape)} dtype={t.dtype}')

def decode_and_render(pipeline, slat, renderer, extrinsics, intrinsics):
    global _DIAG_DONE, _DIAG_HOOKS
    diag = not _DIAG_DONE

    slat_leaf_feats = slat.feats.detach().requires_grad_(True)
    slat_for_decode = sp.SparseTensor(feats=slat_leaf_feats, coords=slat.coords)

    decoded  = pipeline.decode_slat(slat_for_decode, ['gaussian'])
    gaussian = decoded['gaussian'][0]

    if diag:
        print(f'  [G0] slat_leaf_feats:  {_gfn(slat_leaf_feats)}')
        print(f'  [G1] features_dc:      {_gfn(gaussian._features_dc)}')

    # Sanitize ALL Gaussian attrs out-of-place — in-place nan_to_num_ would break
    # saved activations in the backward pass through the decoder.
    # NaN in _xyz gives NaN projection Jacobians → NaN 2D covariance → rasterizer
    # allocates INT_MAX tiles (~66977 TiB) and crashes immediately.
    for _attr in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc']:
        _t = getattr(gaussian, _attr, None)
        if _t is not None:
            setattr(gaussian, _attr, torch.nan_to_num(_t, nan=0.0, posinf=0.0, neginf=0.0))
    gaussian._scaling = gaussian._scaling.clamp(-5.0, 3.0)

    # Fix zero-norm quaternions out-of-place using torch.where (differentiable).
    # If _rotation ≈ -rots_bias, the effective norm cancels to zero → NaN rotation matrix.
    if hasattr(gaussian, 'rots_bias'):
        rot_eff   = gaussian._rotation + gaussian.rots_bias[None, :]
        zero_mask = (rot_eff.norm(dim=-1, keepdim=True) < 1e-6).expand_as(gaussian._rotation)
        gaussian._rotation = torch.where(zero_mask, torch.zeros_like(gaussian._rotation),
                                         gaussian._rotation)

    # Sanitize cov3D and patch get_covariance so the rasterizer uses the clean values.
    # (rasterizer calls pc.get_covariance() internally when pipe.compute_cov3D_python=True)
    cov3d = gaussian.get_covariance()
    cov3d = torch.nan_to_num(cov3d, nan=0.0, posinf=1e-4, neginf=0.0).clamp(-1.0, 1.0)
    gaussian.get_covariance = lambda *a, **kw: cov3d

    result = renderer.render(gaussian, extrinsics, intrinsics)
    color  = result['color']   # (3, H, W), bg already composited with bg_color=(1,1,1)

    if diag:
        print(f'  [G3] rendered color:   {_gfn(color)}')

        def _bhook(name):
            def _h(g):
                mx = g.abs().max().item()
                print(f'  [BWDHOOK] {name}: dtype={g.dtype} max={mx:.3e}  '
                      f'mean={g.abs().mean().item():.3e}  {"<<ZERO!" if mx == 0.0 else "ok"}')
            return _h

        _DIAG_HOOKS.clear()
        _DIAG_HOOKS.append(color.register_hook(_bhook('color')))
        _DIAG_HOOKS.append(gaussian._features_dc.register_hook(_bhook('features_dc')))
        _DIAG_HOOKS.append(slat_leaf_feats.register_hook(_bhook('slat_leaf_feats')))

    return color, slat_leaf_feats


def find_latest_ckpt():
    ckpts = sorted(CKPT_DIR.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _DIAG_DONE
    frame_list = list(range(1, N_FRAMES + 1, FRAME_STRIDE))
    print(f'=== Step 8 Training (Gaussian) ===')
    print(f'  steps={STEPS}  frame_stride={FRAME_STRIDE} ({len(frame_list)} frames/epoch)'
          f'  CFG=disabled  LR={LR}  epochs={EPOCHS}')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\nLoading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Fixed voxel structure (G_S) from frame 75 ─────────────────────────────
    print('\nSampling voxel structure (G_S, frame 75)...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}')

    keep_models = {'slat_flow_model', 'slat_decoder_gs'}
    for name in list(pipeline.models.keys()):
        if name in keep_models:
            continue
        try:
            pipeline.models[name].cpu()
        except Exception:
            pass
        del pipeline.models[name]
    torch.cuda.empty_cache()
    print(f'  Kept models on GPU: {sorted(keep_models)}')

    # Convert Gaussian decoder to fp32 to prevent fp16 overflow → NaN scales → OOM.
    # Also patch self.dtype so the base-class "h = h.type(self.dtype)" line doesn't
    # cast activations back to fp16 mid-forward.
    gs_decoder = pipeline.models['slat_decoder_gs']
    gs_decoder.convert_to_fp32()
    gs_decoder.dtype = torch.float32
    print(f'  Gaussian decoder blocks → fp32')

    # ── Gaussian renderer ──────────────────────────────────────────────────────
    # near=0.8, far=1.6 match phase7 values confirmed to work with Gaussian decoder.
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = RENDER_RES
    renderer.rendering_options.bg_color   = (1.0, 1.0, 1.0)
    renderer.rendering_options.near       = 0.8
    renderer.rendering_options.far        = 1.6

    # Camera: front view matching GT orientation (confirmed in infer_gaussian.py).
    # eye=(0,0,2) + up=(0,1,0) gives same visible orientation as MeshRenderer's
    # confirmed extrinsics. DO NOT change without re-running orientation tests.
    fov        = torch.deg2rad(torch.tensor(40.)).to(DEVICE)
    eye        = torch.tensor([0., 0., 2.]).to(DEVICE)
    tgt        = torch.zeros(3).to(DEVICE)
    up         = torch.tensor([0., 1., 0.]).to(DEVICE)
    EXTRINSICS = u3d.extrinsics_look_at(eye, tgt, up)
    INTRINSICS = u3d.intrinsics_from_fov_xy(fov, fov)

    # ── LoRA + optimizer ──────────────────────────────────────────────────────
    print('Inserting LoRA...')
    lora_blocks, alpha = insert_lora(flow_model, rank=LORA_RANK)
    lora_blocks = lora_blocks.to(DEVICE)
    alpha       = nn.Parameter(alpha.data.to(DEVICE))
    optimizer   = torch.optim.Adam(trainable_params(lora_blocks, alpha), lr=LR)
    print(f'  Trainable: {count_trainable(lora_blocks, alpha):,}')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch  = 1
    loss_history = []
    latest_ckpt  = find_latest_ckpt()
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha       = nn.Parameter(ckpt['alpha'].to(DEVICE))
        optimizer   = torch.optim.Adam(trainable_params(lora_blocks, alpha), lr=LR)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        log_path = RESULTS_DIR / 'loss_history_gs.json'
        if log_path.exists():
            with open(log_path) as f:
                loss_history = json.load(f)
        print(f'  Resumed from {latest_ckpt.name}'
              f'  (epoch {ckpt["epoch"]}, avg_loss={ckpt["avg_loss"]:.5f},'
              f'  alpha={alpha.item():.4f})')
    else:
        print('  No checkpoint found — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already complete.')
        return

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    print('Loading DINOv2...')
    dino = load_dino(DEVICE)

    print(f'\nPre-encoding {N_FRAMES} frames...')
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

    # ── FrameWeightProjector (frozen) ─────────────────────────────────────────
    projector = FrameWeightProjector(k=K).to(DEVICE)
    for p in projector.parameters():
        p.requires_grad_(False)

    # ── Voxel-to-token alignment (Step 5, computed once at frame 75) ──────────
    print('Computing voxel_to_token alignment (frame 75)...')
    tok75       = {75: {k: v.to(DEVICE) for k, v in all_tokens[75].items()}}
    K_hat_75, _ = run_mcfm('v2', tok75, [75], 75, torch.ones(1, device=DEVICE))
    cond_75     = K_hat_75.unsqueeze(0)
    torch.manual_seed(NOISE_SEED)
    sx75 = sp.SparseTensor(
        feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
        coords=coords,
    )
    with torch.no_grad():
        _, _, voxel_to_token = extract_alignment(flow_model, sx75, cond_75, DEVICE)
    print(f'  N_vox_ca: {voxel_to_token.shape[0]}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════════════════
    t0_train = time.time()
    flow_model.train()
    print(f'\nTraining epochs {start_epoch}→{EPOCHS}  '
          f'({len(frame_list)} frames/epoch, {STEPS} steps/frame)')

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_loss  = 0.0
        epoch_skips = 0
        t_epoch     = time.time()

        for frame_num in frame_list:
            t_val     = (frame_num - 1) / (N_FRAMES - 1)
            frame_idx = t_to_frame_idx(t_val)
            win_idx   = get_window_indices(frame_idx, K)

            tokens_gpu = {idx: {k: v.to(DEVICE) for k, v in all_tokens[idx].items()}
                          for idx in win_idx}

            lambda_vec, _ = get_frame_weights(t_val, frame_idx, win_idx, tokens_gpu, projector)
            K_hat, _      = run_mcfm('v2', tokens_gpu, win_idx, frame_idx, lambda_vec)
            cond_gl       = K_hat.unsqueeze(0)

            K_pooled = torch.stack([tokens_gpu[i]['tokens']
                                    for i in dict.fromkeys(win_idx)]).mean(0)

            # Step 6: build E and pass to all denoising steps via dual_path_ctx
            E = build_enhancement_matrix(voxel_to_token, lambda_vec, k=K, N_ctx=1374).to(DEVICE)
            del tokens_gpu, lambda_vec  # keep E — needed for all denoising steps below

            torch.manual_seed(NOISE_SEED + epoch * 200 + frame_num)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )

            # ── Forward ───────────────────────────────────────────────────────
            optimizer.zero_grad()

            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E=E
            )
            x0_val = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha, E=E, require_grad=False
            )
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp

            # ── Decode + render + backward ────────────────────────────────────
            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None:
                _flow_ref.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            try:
                color, slat_leaf_feats = decode_and_render(
                    pipeline, slat_val, renderer, EXTRINSICS, INTRINSICS
                )

                gc.collect()
                torch.cuda.empty_cache()

                gt   = load_gt_frame(frame_idx)
                loss = F.mse_loss(color, gt)
                first_frame_diag = not _DIAG_DONE
                if first_frame_diag:
                    print(f'  [G4] loss: {_gfn(loss)}  val={loss.item():.5f}')
                (loss * LOSS_SCALE).backward()
                if slat_leaf_feats.grad is not None:
                    slat_leaf_feats.grad.div_(LOSS_SCALE)
                if first_frame_diag:
                    g = slat_leaf_feats.grad
                    print(f'  [G5] slat_leaf_feats.grad: '
                          f'{"None" if g is None else f"dtype={g.dtype} max={g.abs().max().item():.3e} mean={g.abs().mean().item():.3e}"}')
                    _DIAG_DONE = True
            finally:
                if _flow_ref is not None:
                    _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render produced no gradient on slat_leaf_feats')
            grad_slat_max = slat_leaf_feats.grad.abs().max().item()
            if grad_slat_max < 1e-30:
                print(f'  WARNING e{epoch:02d} f{frame_num:03d}: slat_leaf_feats.grad is ALL ZERO — LoRA will not update')
            grad_slat = slat_leaf_feats.grad.detach()

            x0_raw = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha, E=E, require_grad=True
            )
            slat = normalize_slat(x0_raw)
            torch.autograd.backward(slat.feats, grad_slat)
            torch.nn.utils.clip_grad_norm_(trainable_params(lora_blocks, alpha), GRAD_CLIP)
            optimizer.step()

            loss_val    = loss.item()
            epoch_loss += loss_val
            loss_history.append({
                'epoch': epoch, 'frame': frame_num,
                'loss': loss_val, 'alpha': alpha.item(),
            })

            if frame_num == frame_list[0] or frame_num % (10 * FRAME_STRIDE) == 0:
                print(f'  e{epoch:02d} f{frame_num:03d}  loss={loss_val:.5f}'
                      f'  alpha={alpha.item():.4f}')

            del color, slat_leaf_feats, gt, loss
            del x0_raw, slat, grad_slat
            del x_prefix, slat_val, cond_gl, K_pooled, E
            _DIAG_HOOKS.clear()
            gc.collect()
            torch.cuda.empty_cache()

        n_trained = len(frame_list) - epoch_skips
        avg_loss  = epoch_loss / max(n_trained, 1)
        elapsed   = time.time() - t_epoch
        print(f'Epoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.5f}'
              f'  alpha={alpha.item():.4f}  skips={epoch_skips}  t={elapsed:.0f}s')

        ckpt_path = CKPT_DIR / f'lora_e{epoch:03d}.pt'
        torch.save({
            'epoch':      epoch,
            'avg_loss':   avg_loss,
            'alpha':      alpha.detach().cpu(),
            'lora_state': lora_blocks.state_dict(),
            'optimizer':  optimizer.state_dict(),
        }, ckpt_path)
        print(f'  Saved: {ckpt_path}')

        with open(RESULTS_DIR / 'loss_history_gs.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    total_min = (time.time() - t0_train) / 60
    print(f'\nDone. Total training time: {total_min:.1f} min')
    print(f'Final alpha: {alpha.item():.4f}')


if __name__ == '__main__':
    main()
