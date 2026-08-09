"""
Step 6.5 — LoRA training loop.

For each of the 150 GT frames:
  1. Steps 1-4: get K̂ (blended) + K_pooled (temporal mean of individual frame tokens)
  2. Get fixed voxel structure (from G_S, computed once and reused)
  3. Run G_L full denoising (25 steps) with dual-path attention (LoRA active)
  4. Decode SLaT → mesh via TRELLIS decoder
  5. Render mesh with nvdiffrast from front-view camera
  6. Loss = MSE(rendered, GT_frame_t)
  7. Backprop through LoRA params + alpha only (TRELLIS frozen)
  8. Adam step

Run from this directory:
  cd .../step6_5_lora
  SPCONV_ALGO=native ATTN_BACKEND=xformers python train.py 2>&1 | tee ../results/train_step6_5.log
"""

import os, sys, math
os.environ['SPCONV_ALGO']        = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']           = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']    = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.renderers.mesh_renderer import render as mesh_render
import nvdiffrast.torch as dr

from step1_input_prep.input_prep       import prepare_input, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from step6_attn_enhancement.enhancement import build_enhancement_matrix
from lora                               import insert_lora, trainable_params, count_trainable
from ray_attention                      import dual_path_ctx

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_DIR      = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                   '/outputs/teapot_lava_kling_premium'
                   '/teapot_lava_kling_premium_front/all_frames_150')
CKPT_DIR    = Path('../results/lora_ckpts')
CKPT_DIR.mkdir(exist_ok=True)

K           = 3        # temporal half-window
LORA_RANK   = 4
LR          = 1e-4
EPOCHS      = 10
NOISE_SEED  = 42
N_STEPS     = 25       # G_L denoising steps
DEVICE      = torch.device('cuda')

# ── Camera for front-view rendering ───────────────────────────────────────────
# Intrinsics: 518×518 output (match GT frame resolution)
RENDER_H, RENDER_W = 518, 518
FOV_DEG     = 45.0
fx = fy     = RENDER_W / (2 * math.tan(math.radians(FOV_DEG / 2)))
cx, cy      = RENDER_W / 2, RENDER_H / 2
INTRINSICS  = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=torch.float32, device=DEVICE)

# Front-view extrinsics: camera 1m in front of object, looking at origin
_cam_pos    = torch.tensor([0, 0, 1.5], dtype=torch.float32)
_look_at    = torch.tensor([0, 0, 0],   dtype=torch.float32)
_up         = torch.tensor([0, 1, 0],   dtype=torch.float32)
_z          = F.normalize(_cam_pos - _look_at, dim=0)
_x          = F.normalize(torch.cross(_up, _z), dim=0)
_y          = torch.cross(_z, _x)
EXTRINSICS  = torch.eye(4, dtype=torch.float32)
EXTRINSICS[:3, :3] = torch.stack([_x, _y, _z], dim=1)
EXTRINSICS[:3,  3] = _cam_pos
EXTRINSICS  = EXTRINSICS.to(DEVICE)


def load_gt(frame_idx: int) -> torch.Tensor:
    """Load GT frame as (1, H, W, 3) float32 tensor in [0,1]."""
    path = GT_DIR / f'frame_{frame_idx:04d}.png'
    img  = Image.open(path).convert('RGB').resize((RENDER_W, RENDER_H), Image.LANCZOS)
    t    = torch.from_numpy(np.array(img)).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)           # (1, H, W, 3)


def compute_K_pooled(tokens: dict, window_indices: list) -> torch.Tensor:
    """Temporal mean of individual frame DINOv2 tokens. Shape: (1374, 1024)."""
    stack = torch.stack([tokens[idx]['tokens'] for idx in dict.fromkeys(window_indices)])
    return stack.mean(dim=0)                    # (1374, 1024)


def run_denoising(pipeline, flow_model, sparse_x, cond_gl, K_pooled,
                  lora_blocks, alpha, device):
    """
    Run G_L denoising for N_STEPS with dual-path LoRA attention.
    Returns: final SparseTensor (denoised SLaT).
    """
    # Get the scheduler from the pipeline's slat sampler
    sampler  = pipeline.slat_sampler
    timesteps = sampler.get_timesteps(N_STEPS)

    x = sparse_x
    with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha):
        for t_val in timesteps:
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32)
            with torch.no_grad():
                # Predict noise (frozen TRELLIS path) — only LoRA grads matter
                pass
            # LoRA-active forward
            noise_pred = flow_model(x, t_tensor, cond_gl)
            x = sampler.step(noise_pred, t_val, x)

    return x


def decode_and_render(pipeline, slat, ctx_gl):
    """
    Decode SLaT to mesh and render from front view with nvdiffrast.
    Returns: rendered image tensor (1, H, W, 3) float32 in [0,1].
    """
    decoded = pipeline.decode_slat(slat, ['mesh'])
    mesh    = decoded['mesh'][0]

    # mesh_render from TRELLIS renderers
    render_result = mesh_render(
        mesh,
        extrinsics=EXTRINSICS.unsqueeze(0),
        intrinsics=INTRINSICS.unsqueeze(0),
        height=RENDER_H,
        width=RENDER_W,
        ctx=ctx_gl,
    )
    return render_result['color']               # (1, H, W, 3)


def main():
    print('Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    flow_model.train()

    # Insert LoRA, freeze TRELLIS
    lora_blocks, alpha = insert_lora(flow_model, rank=LORA_RANK)
    lora_blocks = lora_blocks.to(DEVICE)
    alpha       = nn.Parameter(alpha.data.to(DEVICE))  # .to() strips leaf; reconstruct properly

    n_trainable = count_trainable(lora_blocks, alpha)
    print(f'Trainable params: {n_trainable:,}  ({n_trainable/1e6:.3f}M)')
    print(f'TRELLIS frozen:   {sum(1 for p in flow_model.parameters() if not p.requires_grad):,} params')

    # Optimizer on LoRA only
    optimizer = torch.optim.Adam(trainable_params(lora_blocks, alpha), lr=LR)

    # DINOv2 encoder (frozen)
    dino = load_dino(DEVICE)

    # nvdiffrast GL context for rendering
    ctx_gl = dr.RasterizeGLContext()

    # ── Get fixed voxel structure (G_S once on frame 75) ─────────────────────
    print('Getting fixed voxel structure...')
    _img75 = Image.open(GT_DIR / 'frame_0075.png').convert('RGB')
    _cond  = pipeline.get_cond([_img75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(_cond, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}')

    # ── Training loop ─────────────────────────────────────────────────────────
    frame_indices = list(range(1, N_FRAMES + 1))

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for frame_idx in frame_indices:
            t_val = (frame_idx - 1) / (N_FRAMES - 1)   # 0.0 → 1.0

            # Steps 1-4
            _, window_indices, window_frames, _ = prepare_input(t=t_val, k=K)
            tokens      = encode_frames(window_indices, window_frames, dino, DEVICE)
            projector   = FrameWeightProjector(k=K).to(DEVICE)
            lambda_vec, _ = get_frame_weights(t_val, frame_idx, window_indices,
                                               tokens, projector)
            K_hat, _    = run_mcfm('v1', tokens, window_indices, frame_idx, lambda_vec)
            cond_gl     = K_hat.unsqueeze(0).to(DEVICE)

            # Temporal pool (individual frame mean, not MCFM blend)
            K_pooled = compute_K_pooled(tokens, window_indices).to(DEVICE)

            # Build SparseTensor (same coords every frame; noise resampled)
            torch.manual_seed(NOISE_SEED + frame_idx)
            noise    = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
            sparse_x = sp.SparseTensor(feats=noise, coords=coords)

            # G_L denoising with dual-path LoRA
            optimizer.zero_grad()
            slat = run_denoising(pipeline, flow_model, sparse_x, cond_gl,
                                  K_pooled, lora_blocks, alpha, DEVICE)

            # Render
            rendered = decode_and_render(pipeline, slat, ctx_gl)  # (1, H, W, 3)

            # GT
            gt = load_gt(frame_idx)   # (1, H, W, 3)

            # Loss
            loss = F.mse_loss(rendered, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f'  epoch {epoch+1}/{EPOCHS}  frame {frame_idx:3d}  loss={loss.item():.6f}')

        avg = epoch_loss / N_FRAMES
        print(f'Epoch {epoch+1}/{EPOCHS}  avg_loss={avg:.6f}  alpha={alpha.item():.4f}')

        # Checkpoint
        ckpt_path = CKPT_DIR / f'lora_epoch{epoch+1:03d}.pt'
        torch.save({
            'lora_blocks': lora_blocks.state_dict(),
            'alpha':       alpha.detach(),
            'epoch':       epoch + 1,
            'avg_loss':    avg,
        }, ckpt_path)
        print(f'  Saved: {ckpt_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
