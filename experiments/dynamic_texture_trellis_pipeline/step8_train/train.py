"""
Step 8: Training loop.

Speed settings applied:
  A. No CFG during training — only cond_gl for all denoising steps
  B. 25 denoising steps (24 no_grad + 1 grad)
  C. FRAME_STRIDE=1 (all 150 frames per epoch)

Resume: automatically resumes from latest checkpoint in results/lora_ckpts/.

Gradient path:
  loss ← MSE ← rendered_color ← MeshRenderer ← vertex_attrs
       ← SLatMeshDecoder ← SLaT_x0 (last step, grad) ← LoRA params + alpha

Run from this directory:
  cd .../step8_train
  SPCONV_ALGO=native ATTN_BACKEND=xformers python train.py 2>&1 | tee ../results/train_step8.log
"""

import os, sys, math, json, time, gc
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
# Decoder densifies to 64^3 x 256 x fp32 (~4 GB intermediates). After 25 denoising
# steps the allocator has ~1 GB of freed-but-fragmented blocks it can't coalesce
# into the 768 MB flexicubes needs. expandable_segments returns those blocks to
# CUDA's native allocator so they can be re-issued as one contiguous allocation.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.renderers import MeshRenderer

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
LATENT_NPZ   = ('/net/projects/ranalab/rajhansini/TRELLIS/data'
                '/dynamic_sequences/trellis_seq/frame_0001/latent.npz')
RESULTS_DIR  = Path('../results')
CKPT_DIR     = RESULTS_DIR / 'lora_ckpts'
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

# dec_mesh runs in fp16 internally; the gradient cast to fp16 underflows
# (fp16 min-normal ≈ 6e-5, but ∂loss/∂slat ≈ 1e-6 after the decoder).
# Scale loss up before backward so the fp16 gradient stays non-zero,
# then divide slat_leaf_feats.grad by the same factor afterwards.
LOSS_SCALE   = 4096.0

STEPS        = 25
RESCALE_T    = 3.0
FRAME_STRIDE = 1       # all 150 frames per epoch

# SLaT normalization (from pipeline.json)
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

# Precompute timestep pairs once (for current STEPS)
_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

# Camera (normalized intrinsics — confirmed in verify_step7)
_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                            dtype=torch.float32, device=DEVICE)
# TRELLIS canonical front view: eye=(0,0,2), up=(0,1,0) [Y-up, looking from +Z].
# phase7 GaussianRenderer: [[1,0,0,0],[0,-1,0,0],[0,0,-1,2],[0,0,0,1]].
# nvdiffrast uses opposite Y convention, so flip cam_y row.
# cam_z = 2 - world_z ∈ [1.5, 2.5] for verts in [-0.5,0.5]^3 → covered by near=0.5, far=3.0.
EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32, device=DEVICE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt_frame(frame_idx: int) -> torch.Tensor:
    img = load_frame(frame_idx).resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E):
    """
    Run all Euler steps except the final one with no grad.
    Returns x_{t_last} used as input to the final denoise step.
    """
    x = noise_sp

    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled, lora_blocks, alpha, E, require_grad):
    """
    Run only the final Euler step.
    If require_grad=False, used to get decode/render gradient wrt SLaT value.
    If require_grad=True, used to propagate that gradient into LoRA params.
    """
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


_DIAG_DONE  = False  # print chain diagnostics once only
_DIAG_HOOKS = []    # keep hook handles so they aren't GC'd

def _gfn(t):
    if t is None: return 'None'
    return (f'req={t.requires_grad} '
            f'fn={type(t.grad_fn).__name__ if t.grad_fn else "None"} '
            f'shape={tuple(t.shape)} dtype={t.dtype}')

def decode_and_render(pipeline, slat, renderer):
    global _DIAG_DONE, _DIAG_HOOKS
    diag = not _DIAG_DONE

    slat_leaf_feats = slat.feats.detach().requires_grad_(True)
    slat_for_decode = sp.SparseTensor(feats=slat_leaf_feats, coords=slat.coords)

    if diag:
        print(f'  [G0] slat_leaf_feats: {_gfn(slat_leaf_feats)}')

    decoded = pipeline.decode_slat(slat_for_decode, ['mesh'])
    mesh    = decoded['mesh'][0]

    if diag:
        print(f'  [M1] mesh.vertices:     {_gfn(mesh.vertices)}')
        print(f'  [M2] mesh.vertex_attrs: {_gfn(mesh.vertex_attrs)}')

    result = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color', 'mask'])
    mask   = result['mask'].unsqueeze(0)                # (1, H, W)
    color  = result['color'] * mask + (1.0 - mask)     # composite over white, (3, H, W)

    if diag:
        print(f'  [M3] rendered color: {_gfn(color)}')

        def _bhook(name):
            def _h(g):
                mx = g.abs().max().item()
                print(f'  [BWDHOOK] {name}: dtype={g.dtype} max={mx:.3e}  '
                      f'mean={g.abs().mean().item():.3e}  {"<<ZERO!" if mx == 0.0 else "ok"}')
            return _h

        _DIAG_HOOKS.clear()
        _DIAG_HOOKS.append(color.register_hook(_bhook('color')))
        _DIAG_HOOKS.append(mesh.vertex_attrs.register_hook(_bhook('vertex_attrs')))
        _DIAG_HOOKS.append(slat_leaf_feats.register_hook(_bhook('slat_leaf_feats')))

    return color, slat_leaf_feats


def find_latest_ckpt():
    ckpts = sorted(CKPT_DIR.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _DIAG_DONE
    frame_list = list(range(1, N_FRAMES + 1, FRAME_STRIDE))
    print(f'=== Step 8 Training ===')
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

    # Keep only models needed for step-8 training on GPU.
    # Other pipeline models consume significant VRAM and can trigger OOM in mesh decode.
    keep_models = {'slat_flow_model', 'slat_decoder_mesh'}
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
        # Reload existing loss log if present
        log_path = RESULTS_DIR / 'loss_history.json'
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

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1},
    )

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    print('Loading DINOv2...')
    dino = load_dino(DEVICE)

    # ── Pre-encode all frames (cached on CPU) ─────────────────────────────────
    print(f'\nPre-encoding {N_FRAMES} frames...')
    t0 = time.time()
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    print(f'  Done in {time.time()-t0:.1f}s')
    # DINO is only needed for pre-encoding; free it before training to save VRAM.
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

            # Tokens for window (GPU)
            tokens_gpu = {idx: {k: v.to(DEVICE) for k, v in all_tokens[idx].items()}
                          for idx in win_idx}

            # Steps 3-4: weights + MCFM
            lambda_vec, _ = get_frame_weights(t_val, frame_idx, win_idx, tokens_gpu, projector)
            K_hat, _      = run_mcfm('v2', tokens_gpu, win_idx, frame_idx, lambda_vec)
            cond_gl       = K_hat.unsqueeze(0)

            K_pooled = torch.stack([tokens_gpu[i]['tokens']
                                    for i in win_idx]).mean(0)

            # Step 6: build E and pass to all denoising steps via dual_path_ctx
            E = build_enhancement_matrix(voxel_to_token, N_ctx=1374).to(DEVICE)
            del tokens_gpu, lambda_vec  # keep E — needed for all denoising steps below

            # Noise
            torch.manual_seed(NOISE_SEED + epoch * 200 + frame_num)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )

            # ── Forward ───────────────────────────────────────────────────────
            optimizer.zero_grad()

            # Memory-friendly training:
            # 1) Decode/render gradient wrt slat using no-grad denoise.
            # 2) Recompute final denoise step with grad and inject that slat-grad.
            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E=E
            )
            x0_val = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, alpha, E=E, require_grad=False
            )
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp    # no longer needed; keep x_prefix for grad recompute

            # ── Decode + render + backward ────────────────────────────────────
            # Swap: flow_model → CPU, decoder → GPU.
            # ── Decode + render + backward (flow_model on CPU throughout) ────────
            # Offloading flow_model frees memory for decoder's dense 64^3 activations.
            # expandable_segments:True (set at top) handles remaining fragmentation.
            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None:
                _flow_ref.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                color, slat_leaf_feats = decode_and_render(pipeline, slat_val, renderer)

                # Clear PyTorch cache after forward so backward gets a clean pool.
                gc.collect()
                torch.cuda.empty_cache()

                # ── Loss + backward ───────────────────────────────────────────
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
                # Restore flow_model to GPU before grad injection step.
                if _flow_ref is not None:
                    _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render produced no gradient on slat_leaf_feats')
            grad_slat_max = slat_leaf_feats.grad.abs().max().item()
            if grad_slat_max < 1e-30:
                print(f'  WARNING e{epoch:02d} f{frame_num:03d}: slat_leaf_feats.grad is ALL ZERO — LoRA will not update')
            grad_slat = slat_leaf_feats.grad.detach()

            # Recompute only final denoise step with grad enabled, then inject
            # decoder-render gradient into this differentiable SLaT graph.
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

            # Free all frame-local GPU tensors so they don't compete with the next
            # frame's decode_and_render forward.  Python loop variables persist in
            # function scope; without explicit del they stay live until reassigned,
            # and PyTorch's allocator keeps them in its cache across iterations.
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

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_path = CKPT_DIR / f'lora_e{epoch:03d}.pt'
        torch.save({
            'epoch':      epoch,
            'avg_loss':   avg_loss,
            'alpha':      alpha.detach().cpu(),
            'lora_state': lora_blocks.state_dict(),
            'optimizer':  optimizer.state_dict(),
        }, ckpt_path)
        print(f'  Saved: {ckpt_path}')

        # ── Loss log ─────────────────────────────────────────────────────────
        with open(RESULTS_DIR / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

    total_min = (time.time() - t0_train) / 60
    print(f'\nDone. Total training time: {total_min:.1f} min')
    print(f'Final alpha: {alpha.item():.4f}')


if __name__ == '__main__':
    main()
