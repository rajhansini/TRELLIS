"""
Step 10 — LoRA Inference: Render all 150 frames with trained LoRA checkpoint.

Loads a trained step07b checkpoint, runs the full pipeline (denoise → decode →
render) for every frame, saves PNGs and a side-by-side comparison video.

BACKWARD COMPATIBLE:
  - Does NOT modify step07b, checkpoints, or any training scripts
  - Writes ONLY to results_mcfm_{mode}_noenh_lora_seed6/renders_e{epoch:03d}/

Usage:
  python step10_render_lora_inference.py --mode v3_C
  python step10_render_lora_inference.py --mode v2_C --ckpt_epoch 50
  python step10_render_lora_inference.py --mode v3_D --frame_stride 10  # quick preview

Outputs in results_mcfm_{mode}_noenh_lora_seed6/renders_e{epoch:03d}/:
  rendered/frame_{i:04d}.png    — LoRA rendered frame
  compare/frame_{i:04d}.png     — GT (left) | rendered (right) side by side
  comparison_video.mp4          — full comparison video
  render_inference.log          — full log
"""

import sys, os, argparse as _ap, subprocess
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
_pre.add_argument('--mode',        type=str, default='v3_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_pre.add_argument('--ckpt_epoch',  type=int, default=-1,
                  help='Checkpoint epoch to load (-1 = latest)')
_pre.add_argument('--frame_stride',type=int, default=1,
                  help='1 = all 150 frames, 10 = every 10th frame (quick preview)')
_PRE, _ = _pre.parse_known_args()

_TRAIN_RESULTS = _HERE / f'results_mcfm_{_PRE.mode}_selfcons_lora_seed6'
_CKPT_DIR      = _TRAIN_RESULTS / 'lora_ckpts'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()


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
from step6_5_lora.lora_v2   import build_lora_blocks, freeze_trellis, gate0_verify
from step6_5_lora.dual_path_v2 import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render,
                                               RENDER_RES, EXTRINSICS, INTRINSICS,
                                               SLAT_MEAN, SLAT_STD)

# ── Constants (must match step07b exactly) ─────────────────────────────────────
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED    = 'microsoft/TRELLIS-image-large'
DEVICE        = torch.device('cuda')
N_FRAMES      = 150
STRUCT_SEED   = 42
FIXED_SEED    = 6
STEPS         = 25
RESCALE_T     = 3.0
LORA_RANK     = 4

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_window(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    else:
        return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(win: list) -> torch.Tensor:
    return torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)


def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def blend(mode: str, raw_tokens: dict, frame_idx: int):
    win      = get_window(frame_idx, mode)
    lam      = get_lambda_vec(win)
    tok_d    = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn  = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    K_pooled = torch.stack([raw_tokens[i].to(DEVICE) for i in win], dim=0).mean(0)
    return K_hat, K_pooled, win


def denoise_full(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha):
    """All 25 denoising steps, no grad."""
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha,
                                   enhance_bias=None):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def find_ckpt(ckpt_epoch: int) -> Path:
    if ckpt_epoch == -1:
        ckpts = sorted(_CKPT_DIR.glob('lora_e*.pt'))
        if not ckpts:
            raise FileNotFoundError(f'No checkpoints in {_CKPT_DIR}')
        return ckpts[-1]
    else:
        p = _CKPT_DIR / f'lora_e{ckpt_epoch:03d}.pt'
        if not p.exists():
            raise FileNotFoundError(f'Checkpoint not found: {p}')
        return p


def save_png(tensor_chw: torch.Tensor, path: Path):
    """Save (3, H, W) float [0,1] tensor as PNG."""
    arr = (tensor_chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def make_comparison(gt_path: Path, rendered_tensor: torch.Tensor, out_path: Path):
    """GT (left) | rendered (right) side by side."""
    gt  = Image.open(gt_path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    arr = (rendered_tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rnd = Image.fromarray(arr)
    combined = Image.new('RGB', (RENDER_RES * 2, RENDER_RES))
    combined.paste(gt,  (0, 0))
    combined.paste(rnd, (RENDER_RES, 0))
    combined.save(out_path)


def make_video(compare_dir: Path, out_path: Path, fps: int = 10):
    """Use ffmpeg concat demuxer to create mp4 from comparison PNGs."""
    frames = sorted(compare_dir.glob('frame_*.png'))
    if not frames:
        print('  [VIDEO] no frames found, skipping')
        return
    filelist = out_path.parent / '_ffmpeg_filelist.txt'
    with open(filelist, 'w') as f:
        for p in frames:
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0/fps:.6f}\n")
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', str(filelist),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    filelist.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f'  [VIDEO] ffmpeg error: {result.stderr[-400:]}')
    else:
        print(f'  [VIDEO] saved → {out_path}')


def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode',         type=str, default='v3_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--ckpt_epoch',   type=int, default=-1)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--fps',          type=int, default=10)
    parser.add_argument('--force_alpha',  type=float, default=None,
                        help='Override loaded alpha (e.g. 0.0 to disable LoRA)')
    args = parser.parse_args()
    mode = args.mode

    ckpt_path  = find_ckpt(args.ckpt_epoch)
    ckpt_epoch = int(ckpt_path.stem.replace('lora_e', ''))

    _RENDER_DIR  = _TRAIN_RESULTS / f'renders_e{ckpt_epoch:03d}'
    _RENDERED    = _RENDER_DIR / 'rendered'
    _COMPARE     = _RENDER_DIR / 'compare'
    for d in [_RENDER_DIR, _RENDERED, _COMPARE]:
        d.mkdir(parents=True, exist_ok=True)

    sys.stdout = _Tee(_RENDER_DIR / 'render_inference.log')
    sys.stderr = sys.stdout

    print('=' * 72)
    print('Step 10 — LoRA Inference Rendering')
    print('=' * 72)
    print(f'  mode         : {mode}')
    print(f'  checkpoint   : {ckpt_path.name}  (epoch {ckpt_epoch})')
    print(f'  frame_stride : {args.frame_stride}')
    print(f'  results      : {_RENDER_DIR}')
    print()

    # ── Load pipeline ──────────────────────────────────────────────────────────
    print('[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Build voxel structure (must match training: STRUCT_SEED=42) ────────────
    print('[STRUCT] Building voxel structure (STRUCT_SEED=42)...')
    img_75      = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1,
                                              sampler_params={'steps': 50,
                                                              'cfg_strength': 7.5})
    print(f'  N_vox={coords.shape[0]}  coords_sum={coords.sum().item():.0f}')

    # ── Fixed noise (must match training: FIXED_SEED=6) ───────────────────────
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(coords.shape[0], 8, device=DEVICE)

    # ── Build LoRA + load checkpoint ───────────────────────────────────────────
    print('[LORA] Building LoRA blocks...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK).to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))

    print(f'[CKPT] Loading {ckpt_path.name}...')
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'beta' in ckpt:
        raise RuntimeError('Checkpoint has beta key — belongs to step07 (enhancement). Use step07b checkpoints.')
    if 'mode' in ckpt and ckpt['mode'] != mode:
        raise RuntimeError(f'Checkpoint mode={ckpt["mode"]} but --mode={mode}')
    lora_blocks.load_state_dict(ckpt['lora_state'])
    alpha = nn.Parameter(ckpt['alpha'].to(DEVICE))
    if args.force_alpha is not None:
        alpha = nn.Parameter(torch.tensor(args.force_alpha, device=DEVICE))
        print(f'  epoch={ckpt_epoch}  avg_loss={ckpt["avg_loss"]:.5f}  alpha=FORCED→{alpha.item():.4f}')
    else:
        print(f'  epoch={ckpt_epoch}  avg_loss={ckpt["avg_loss"]:.5f}  alpha={alpha.item():.4f}')

    # Move decoder back to GPU, flow model stays on GPU
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.to(DEVICE)
    flow_model.eval()
    lora_blocks.eval()

    # ── Encode all frames ──────────────────────────────────────────────────────
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
    dino_model.cpu()
    torch.cuda.empty_cache()
    print('  Done.')

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer  = make_renderer()
    frame_list = list(range(1, N_FRAMES + 1, args.frame_stride))
    print(f'\n[RENDER] Rendering {len(frame_list)} frames...')

    first_diag = True
    t0 = time.time()

    for frame_i in frame_list:
        K_hat, K_pooled, win = blend(mode, raw_tokens, frame_i)
        cond_gl = K_hat.unsqueeze(0)

        noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)

        # Full 25-step denoise, no grad
        _flow_ref = pipeline.models.get('slat_flow_model')
        x0 = denoise_full(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha)
        slat = normalize_slat(x0)

        # Move flow model off GPU for decode
        if _flow_ref is not None: _flow_ref.cpu()
        gc.collect(); torch.cuda.empty_cache()

        color, _ = decode_and_render(pipeline, slat, renderer,
                                     diag=first_diag, device=DEVICE)
        first_diag = False

        if _flow_ref is not None: _flow_ref.to(DEVICE)

        # Save rendered frame
        save_png(color, _RENDERED / f'frame_{frame_i:04d}.png')

        # Save side-by-side comparison
        gt_path = GT_FRAMES_DIR / f'frame_{frame_i:04d}.png'
        make_comparison(gt_path, color, _COMPARE / f'frame_{frame_i:04d}.png')

        elapsed = time.time() - t0
        print(f'  frame {frame_i:3d}/{N_FRAMES}  win={win}  '
              f'elapsed={elapsed:.1f}s')

        del K_hat, K_pooled, cond_gl, noise_sp, x0, slat, color
        gc.collect(); torch.cuda.empty_cache()

    # ── Video ──────────────────────────────────────────────────────────────────
    print('\n[VIDEO] Creating comparison video...')
    make_video(_COMPARE, _RENDER_DIR / 'comparison_video.mp4', fps=args.fps)

    print(f'\n[DONE] {len(frame_list)} frames rendered in {time.time()-t0:.1f}s')
    print(f'  Rendered frames : {_RENDERED}')
    print(f'  Comparisons     : {_COMPARE}')
    print(f'  Video           : {_RENDER_DIR}/comparison_video.mp4')
    print(f'  Log             : {_RENDER_DIR}/render_inference.log')


if __name__ == '__main__':
    main()
