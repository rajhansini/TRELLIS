"""
Phase B/C baseline: simplified pipeline with no LoRA, no enhancement matrix.

Stages:
  1. Input prep  — window = [frame_i, frame_{i+1}]
  2. DINOv2      — encode both frames
  3. Weights     — lambda_curr + lambda_next = 1.0  (manually set)
  4. MCFM v1     — weighted average of the two frames' tokens
  5. Denoise     — vanilla flow model, 25 Euler steps, no LoRA, no E matrix
  6. Decode      — SLaT -> mesh
  7. Render      — nvdiffrast -> PNG

At lambda_curr=1.0, lambda_next=0.0 -> pure vanilla TRELLIS (flickering baseline).

Usage:
  python run.py --lambda_curr 1.0 --lambda_next 0.0 --tag flickering
  python run.py --lambda_curr 0.9 --lambda_next 0.1 --tag blend_01
  python run.py --lambda_curr 0.5 --lambda_next 0.5 --tag blend_05
"""

import os, sys, argparse, math, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
from PIL import Image
import subprocess
from pathlib import Path

# ── repo root on path ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent
_ROOT = _PIPE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp

from step1_input_prep.input_prep   import load_frame, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step4_mcfm.mcfm               import mcfm_v1

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = str(_PIPE / '..' / '..' / '..' /
                  'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
RESULTS_DIR = _PIPE / 'results' / 'phase_b'
DEVICE      = torch.device('cuda')
STEPS       = 25
NOISE_SEED  = 42
RENDER_RES  = 518
RESCALE_T   = 3.0

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_slat(x0: sp.SparseTensor) -> sp.SparseTensor:
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def denoise(flow_model, noise_sp: sp.SparseTensor, cond_gl: torch.Tensor) -> sp.SparseTensor:
    """Vanilla Euler denoising — no LoRA, no E matrix."""
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def blend_tokens(tokens_curr: torch.Tensor, tokens_next: torch.Tensor,
                 lc: float, ln: float) -> torch.Tensor:
    """Weighted average of two frames' tokens -> (1374, 1024)."""
    return lc * tokens_curr + ln * tokens_next


def render_slat(pipeline, slat: sp.SparseTensor, renderer: MeshRenderer):
    """Decode SLaT and render. Returns HxWx3 uint8 numpy array."""
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    mask  = result['mask'].unsqueeze(0)                 # (1, H, W)
    color = result['color'] * mask + (1.0 - mask)      # composite white bg, (3, H, W)
    img   = (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_curr', type=float, default=1.0,
                        help='Weight for current frame tokens (default 1.0 = flickering)')
    parser.add_argument('--lambda_next', type=float, default=0.0,
                        help='Weight for next frame tokens (default 0.0)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Output subfolder tag (default: lc{lambda_curr}_ln{lambda_next})')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to render (default: all 150)')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='First frame index to render (default: 1)')
    args = parser.parse_args()

    lc = args.lambda_curr
    ln = args.lambda_next
    assert abs(lc + ln - 1.0) < 1e-5, f'lambda_curr + lambda_next must equal 1.0, got {lc+ln}'

    tag      = args.tag or f'lc{lc:.2f}_ln{ln:.2f}'
    out_dir  = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    start    = args.start_frame
    n_frames = args.frames or N_FRAMES

    print(f'=== Phase B/C Baseline ===')
    print(f'  lambda_curr={lc}  lambda_next={ln}  tag={tag}')
    print(f'  frames=1..{n_frames}  steps={STEPS}')
    print(f'  output: {out_dir}')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\nLoading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Sample fixed voxel structure from frame 75 ────────────────────────────
    # Must happen before offloading — image_cond_model needs to be on GPU.
    print('Sampling voxel structure from frame 75...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}')

    # Keep only flow model + mesh decoder on GPU
    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── DINOv2 + pre-encode all frames ────────────────────────────────────────
    print(f'\nPre-encoding {N_FRAMES} frames with DINOv2...')
    dino = load_dino(DEVICE)
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

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )

    # ── Per-frame rendering ───────────────────────────────────────────────────
    print(f'\nRendering {n_frames} frames...')
    flow_model.eval()

    for frame_i in range(start, start + n_frames):
        frame_next = min(N_FRAMES, frame_i + 1)

        # Tokens for [i, i+1] on GPU
        tok_curr = all_tokens[frame_i  ]['tokens'].to(DEVICE)   # (1374, 1024)
        tok_next = all_tokens[frame_next]['tokens'].to(DEVICE)

        # Step 3+4: weighted blend
        K_hat   = blend_tokens(tok_curr, tok_next, lc, ln)      # (1374, 1024)
        cond_gl = K_hat.unsqueeze(0)                             # (1, 1374, 1024)

        # Noise (fixed seed per frame for reproducibility)
        torch.manual_seed(NOISE_SEED + frame_i)
        noise_sp = sp.SparseTensor(
            feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
            coords=coords,
        )

        # Step 5: denoise
        x0     = denoise(flow_model, noise_sp, cond_gl)
        slat   = normalize_slat(x0)
        del noise_sp, x0, tok_curr, tok_next

        # Steps 6-7: decode + render
        rendered = render_slat(pipeline, slat, renderer)
        del slat
        torch.cuda.empty_cache()

        # Save
        out_path = out_dir / f'frame_{frame_i:04d}.png'
        Image.fromarray(rendered).save(out_path)

        if frame_i % 10 == 0 or frame_i == 1:
            print(f'  [{frame_i:03d}/{n_frames}] saved {out_path.name}')

    # ── Assemble video ────────────────────────────────────────────────────────
    if n_frames == 1:
        print('Single frame — skipping video assembly.')
        return
    video_path = out_dir / 'render.mp4'
    print(f'\nAssembling video -> {video_path}')
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', '25',
        '-i', str(out_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(video_path)
    ], check=True)
    print(f'Done. Video: {video_path}')


if __name__ == '__main__':
    main()
