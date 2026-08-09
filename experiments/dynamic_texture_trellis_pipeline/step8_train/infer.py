"""
Inference: render animated texture for all 150 frames using trained LoRA.

Loads the latest lora_e*.pt checkpoint, runs the full pipeline for every frame,
saves rendered PNGs and a side-by-side comparison video.

Run from step8_train/:
  SPCONV_ALGO=native ATTN_BACKEND=xformers python infer.py
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
from trellis.renderers import MeshRenderer

from step1_input_prep.input_prep       import N_FRAMES, load_frame, t_to_frame_idx, get_window_indices
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
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
OUT_DIR     = RESULTS_DIR / 'inference_frames'
OUT_DIR.mkdir(parents=True, exist_ok=True)

K          = 3
LORA_RANK  = 4
NOISE_SEED = 42      # fixed for all frames — temporal variation comes from conditioning only
RENDER_RES = 518
STEPS      = 10
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

_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32, device=DEVICE)


def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def denoise_all(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=None):
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
    keep_models = {'slat_flow_model', 'slat_decoder_mesh'}
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

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1},
    )

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

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f'\nRendering {N_FRAMES} frames...')
    flow_model.eval()
    t0 = time.time()
    skipped = 0

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

        torch.manual_seed(NOISE_SEED)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )

        x0   = denoise_all(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha)
        slat = normalize_slat(x0)
        del noise_sp, x0, cond_gl, K_pooled, tokens_gpu, lambda_vec

        _flow_ref = pipeline.models.get('slat_flow_model')
        if _flow_ref is not None:
            _flow_ref.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        if frame_num == 1:
            print(f'  [MEM] f001 after flow.cpu(): {_mem()}')
        try:
            with torch.no_grad():
                decoded = pipeline.decode_slat(slat, ['mesh'])
                mesh    = decoded['mesh'][0]
                result  = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color', 'mask'])
            color = result['color']                     # (3, H, W) float [0,1], black bg
            mask  = result['mask'].unsqueeze(0)         # (1, H, W)
            color = color * mask + (1.0 - mask)         # composite over white
            color_np = (color.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            color_np = color_np.transpose(1, 2, 0)   # (H, W, 3)
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

    print(f'\n  Done. {N_FRAMES - skipped}/{N_FRAMES} frames rendered, {skipped} skipped.')

    # ── Compile videos ────────────────────────────────────────────────────────
    print('Compiling videos...')
    for prefix, label in [('render', 'render'), ('comparison', 'comparison')]:
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
    print(f'  Frames:      {OUT_DIR}/')
    print(f'  Render MP4:  {RESULTS_DIR}/render.mp4')
    print(f'  Comparison:  {RESULTS_DIR}/comparison.mp4')


if __name__ == '__main__':
    main()
