"""
Inference: render animated texture for all 150 frames using trained LoRA.
Uses Gaussian decoder + GaussianRenderer for better texture quality.

Loads the latest lora_e*.pt checkpoint, runs the full pipeline for every frame,
saves rendered PNGs and a side-by-side comparison video.

Run from step8_train/:
  SPCONV_ALGO=native ATTN_BACKEND=xformers python infer_gaussian.py
"""

import os, sys, math, subprocess, time, gc
import numpy as np
import torch
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

from step1_input_prep.input_prep       import N_FRAMES, load_frame, t_to_frame_idx, get_window_indices
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from step6_attn_enhancement.enhancement import build_enhancement_matrix
from step6_5_lora.lora                  import insert_lora, trainable_params
from step6_5_lora.ray_attention         import dual_path_ctx

# ── Config (must match train.py) ──────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
LATENT_NPZ  = ('/net/projects/ranalab/rajhansini/TRELLIS/data'
               '/dynamic_sequences/trellis_seq/frame_0001/latent.npz')
RESULTS_DIR = Path('../results')
CKPT_DIR    = RESULTS_DIR / 'lora_ckpts'
OUT_DIR     = RESULTS_DIR / 'inference_frames_gaussian'
OUT_DIR.mkdir(parents=True, exist_ok=True)

K          = 3
LORA_RANK  = 4
NOISE_SEED = 42
RENDER_RES = 518
STEPS      = 25
RESCALE_T  = 3.0
DEVICE     = torch.device('cuda')

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


def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def denoise_all(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def main():
    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    def _mem(): return f'{torch.cuda.memory_allocated()/1e9:.2f} GB'

    print(f'  [MEM] after pipeline.to(DEVICE): {_mem()}')

    # ── Voxel structure (same as training) ────────────────────────────────────
    print('Sampling voxel structure (frame 75)...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    del cond_struct
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}  [MEM] {_mem()}')

    # Keep only flow model + gaussian decoder on GPU
    keep_models = {'slat_flow_model', 'slat_decoder_gs'}
    for name in list(pipeline.models.keys()):
        if name in keep_models:
            continue
        try:
            pipeline.models[name].cpu()
        except Exception:
            pass
        del pipeline.models[name]
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  [MEM] after model prune: {_mem()}')

    # The Gaussian decoder ships as fp16 but LoRA-modified SLaT features can
    # push it into fp16 overflow → NaN scales → rasterizer allocates INT_MAX
    # tiles per Gaussian → astronomical OOM.
    # Fix: convert blocks to fp32 AND patch self.dtype so the base-class
    # "h = h.type(self.dtype)" line no longer casts activations back to fp16.
    gs_decoder = pipeline.models['slat_decoder_gs']
    gs_decoder.convert_to_fp32()   # converts self.blocks weights to fp32
    gs_decoder.dtype = torch.float32  # prevents activation cast to fp16 in forward
    print(f'  Gaussian decoder blocks → fp32  [MEM] {_mem()}')

    # ── Gaussian renderer ──────────────────────────────────────────────────────
    # near=0.8, far=1.6 match phase7 values proven to work with Gaussian decoder
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = RENDER_RES
    renderer.rendering_options.bg_color   = (1.0, 1.0, 1.0)
    renderer.rendering_options.near       = 0.8
    renderer.rendering_options.far        = 1.6

    # ── Gaussian camera: front view matching GT orientation ────────────────────
    # Gaussian space has front=+Z. eye=(0,0,2) + up=(0,1,0) gives the same
    # visible orientation as the confirmed MeshRenderer camera (test_front9).
    fov      = torch.deg2rad(torch.tensor(40.)).to(DEVICE)
    eye      = torch.tensor([0., 0., 2.]).to(DEVICE)
    tgt      = torch.zeros(3).to(DEVICE)
    up       = torch.tensor([0., 1., 0.]).to(DEVICE)
    EXTRINSICS = u3d.extrinsics_look_at(eye, tgt, up)   # (4, 4)
    INTRINSICS = u3d.intrinsics_from_fov_xy(fov, fov)   # (3, 3)

    # ── LoRA — load latest checkpoint ─────────────────────────────────────────
    print('Loading LoRA checkpoint...')
    lora_blocks, alpha = insert_lora(flow_model, rank=LORA_RANK)
    lora_blocks = lora_blocks.to(DEVICE)
    alpha       = alpha.to(DEVICE)

    latest_ckpt = sorted(CKPT_DIR.glob('lora_e*.pt'))[-1]
    ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
    lora_blocks.load_state_dict(ckpt['lora_state'])
    alpha.data.copy_(ckpt['alpha'].to(DEVICE))
    lora_blocks.eval()
    print(f'  {latest_ckpt.name}  epoch={ckpt["epoch"]}  alpha={alpha.item():.4f}')

    # ── FrameWeightProjector (frozen random, same as training) ────────────────
    projector = FrameWeightProjector(k=K).to(DEVICE)
    for p in projector.parameters():
        p.requires_grad_(False)

    # ── DINOv2 — pre-encode all frames ────────────────────────────────────────
    print(f'Pre-encoding {N_FRAMES} frames...')
    dino = load_dino(DEVICE)
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    del dino
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  [MEM] after del dino: {_mem()}')

    # ── Step 5: voxel-to-token alignment (computed once at frame 75) ─────────
    print('Computing voxel-to-token alignment (frame 75)...')
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
    print(f'  N_vox_ca: {voxel_to_token.shape[0]}  [MEM] {_mem()}')
    del tok75, K_hat_75, cond_75, sx75
    torch.cuda.empty_cache()

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f'\nRendering {N_FRAMES} frames...')
    flow_model.eval()
    t0 = time.time()

    for frame_num in range(1, N_FRAMES + 1):
        t_val     = (frame_num - 1) / (N_FRAMES - 1)
        frame_idx = t_to_frame_idx(t_val)
        win_idx   = get_window_indices(frame_idx, K)

        tokens_gpu = {idx: {k: v.to(DEVICE) for k, v in all_tokens[idx].items()}
                      for idx in win_idx}

        lambda_vec, _ = get_frame_weights(t_val, frame_idx, win_idx, tokens_gpu, projector)
        K_hat, _      = run_mcfm('v2', tokens_gpu, win_idx, frame_idx, lambda_vec)
        cond_gl       = K_hat.unsqueeze(0)
        K_pooled      = torch.stack([tokens_gpu[i]['tokens']
                                     for i in dict.fromkeys(win_idx)]).mean(0)

        # Step 6: build E for this frame (voxel_to_token fixed from frame 75)
        E = build_enhancement_matrix(voxel_to_token, lambda_vec, k=K, N_ctx=1374).to(DEVICE)
        del tokens_gpu, lambda_vec

        torch.manual_seed(NOISE_SEED)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )

        x0   = denoise_all(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha, E=E)
        slat = normalize_slat(x0)
        del noise_sp, x0, cond_gl, K_pooled, E

        _flow_ref = pipeline.models.get('slat_flow_model')
        if _flow_ref is not None:
            _flow_ref.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        if frame_num == 1:
            print(f'  [MEM] f001 after flow.cpu(): {_mem()}')
        try:
            with torch.no_grad():
                decoded  = pipeline.decode_slat(slat, ['gaussian'])
                gaussian = decoded['gaussian'][0]
                if frame_num == 1:
                    print(f'  N_gaussians: {gaussian._xyz.shape[0]}')
                    for attr in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc']:
                        t = getattr(gaussian, attr, None)
                        if t is None: continue
                        bad = (~torch.isfinite(t)).sum().item()
                        print(f'  {attr}: range [{t.min().item():.3f}, {t.max().item():.3f}]  bad={bad}')
                # Sanitize ALL raw parameters (NaN/Inf → 0)
                for attr in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc']:
                    t = getattr(gaussian, attr, None)
                    if t is not None:
                        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
                gaussian._scaling.clamp_(-5.0, 3.0)
                # Fix zero-norm quaternions: _rotation + rots_bias must have nonzero norm.
                # If _rotation ≈ -rots_bias the norm cancels to zero → NaN rotation matrix.
                if hasattr(gaussian, 'rots_bias'):
                    rot_eff = gaussian._rotation + gaussian.rots_bias[None, :]
                    zero_mask = rot_eff.norm(dim=-1) < 1e-6
                    n_fixed = zero_mask.sum().item()
                    if n_fixed > 0:
                        gaussian._rotation[zero_mask] = 0.0  # identity after adding rots_bias
                    if frame_num == 1:
                        print(f'  Zero-norm quats fixed: {n_fixed}')
                # Pre-compute cov3D, sanitize it, then monkey-patch gaussian so the
                # renderer uses our safe values instead of recomputing from scratch.
                # (renderer calls pc.get_covariance() internally — that's why we patch)
                cov3d = gaussian.get_covariance()
                if frame_num == 1:
                    bad_cov = (~torch.isfinite(cov3d)).sum().item()
                    print(f'  cov3D: range [{cov3d.min().item():.3e}, {cov3d.max().item():.3e}]  bad={bad_cov}')
                torch.nan_to_num_(cov3d, nan=0.0, posinf=1e-4, neginf=0.0)
                cov3d.clamp_(-1.0, 1.0)
                gaussian.get_covariance = lambda *a, **kw: cov3d  # patch for renderer
                result = renderer.render(gaussian, EXTRINSICS, INTRINSICS)
            color    = result['color']                      # (3, H, W) float [0,1], bg composited
            color_np = (color.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            color_np = color_np.transpose(1, 2, 0)         # (H, W, 3)
        finally:
            if _flow_ref is not None:
                _flow_ref.to(DEVICE)

        # Rendered frame
        Image.fromarray(color_np).save(OUT_DIR / f'render_{frame_num:04d}.png')

        # Side-by-side: GT | render
        gt_np = np.array(load_frame(frame_num).resize((RENDER_RES, RENDER_RES)))
        sbs   = np.concatenate([gt_np, color_np], axis=1)
        Image.fromarray(sbs).save(OUT_DIR / f'comparison_{frame_num:04d}.png')

        if frame_num == 1 or frame_num % 25 == 0 or frame_num == N_FRAMES:
            elapsed = time.time() - t0
            print(f'  f{frame_num:03d}/{N_FRAMES}  ({elapsed:.0f}s elapsed)')

    print(f'\n  Done. {N_FRAMES} frames rendered.')

    # ── Compile videos ────────────────────────────────────────────────────────
    print('Compiling videos...')
    for prefix, label in [('render', 'render_gaussian'), ('comparison', 'comparison_gaussian')]:
        pattern = str(OUT_DIR / f'{prefix}_%04d.png')
        out_mp4 = str(RESULTS_DIR / f'{label}.mp4')
        cmd = [
            'ffmpeg', '-y', '-framerate', '24',
            '-i', pattern,
            '-c:v', 'mpeg4', '-q:v', '3',
            out_mp4,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f'  Saved: {out_mp4}')
        else:
            print(f'  ffmpeg failed for {prefix}: {result.stderr.decode()[-200:]}')

    print('\nResults:')
    print(f'  Frames:         {OUT_DIR}/')
    print(f'  Render MP4:     {RESULTS_DIR}/render_gaussian.mp4')
    print(f'  Comparison MP4: {RESULTS_DIR}/comparison_gaussian.mp4')


if __name__ == '__main__':
    main()
