"""
Render all 150 frames using a trained LoRA checkpoint, then make a
GT | MCFM | LoRA side-by-side comparison video.

Reuses blend/render logic verbatim from step07c to avoid reimplementing.

Usage:
  python render_lora_all_frames.py --mode v3_C
  python render_lora_all_frames.py --mode v3_C --ckpt lora_e010.pt
"""

import sys, os, gc, argparse, subprocess
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline')

os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step4_mcfm.mcfm        import mcfm_v2, mcfm_v3
from step6_5_lora.lora_v2   import build_lora_blocks, freeze_trellis
from step6_5_lora.dual_path_v2 import dual_path_ctx_v2
from step8_decode_render.decode_render import make_renderer, normalize_slat, decode_and_render

DEVICE      = torch.device('cuda')
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
LORA_RANK   = 4
N_FRAMES    = 150
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'

VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
ENH_DIR = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')

_t_seq  = torch.linspace(1, 0, STEPS + 1).tolist()
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])


# ── Exact copies from step07c ──────────────────────────────────────────────────

def get_window(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(win: list) -> torch.Tensor:
    return torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)


def blend(mode: str, raw_tokens: dict, frame_idx: int):
    win      = get_window(frame_idx, mode)
    lam      = get_lambda_vec(win)
    tok_d    = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn  = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    K_pooled = torch.stack([raw_tokens[i].to(DEVICE) for i in win], dim=0).mean(0)
    return K_hat, K_pooled, win


def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    x   = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0).cpu()


def render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                        raw_tokens, mode, frame_idx, coords, fixed_noise_feats, renderer):
    K_hat, K_pooled, _ = blend(mode, raw_tokens, frame_idx)
    ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl = K_hat.unsqueeze(0)
    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled.to(DEVICE), lora_blocks,
                                   alpha, enhance_bias=None):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    flow_model.cpu()
    del K_hat, K_pooled, cond_gl
    gc.collect(); torch.cuda.empty_cache()
    slat  = normalize_slat(ns)
    del ns
    gc.collect(); torch.cuda.empty_cache()
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    del slat
    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()
    return color.detach().clamp(0, 1)


# ── Video ──────────────────────────────────────────────────────────────────────

def make_video(frames_dir, out_mp4, fps):
    fl = frames_dir / '_list.txt'
    with open(fl, 'w') as f:
        for p in sorted(frames_dir.glob('frame_*.png')):
            f.write(f"file '{p}'\nduration {1.0/fps:.6f}\n")
    r = subprocess.run(
        ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(fl),
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out_mp4)],
        capture_output=True, text=True)
    fl.unlink(missing_ok=True)
    if r.returncode != 0:
        print(f'ffmpeg error: {r.stderr[-300:]}')
    else:
        print(f'Video: {out_mp4}  ({out_mp4.stat().st_size/1e6:.1f} MB)')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  default='v3_C', choices=['v2_C','v2_D','v3_C','v3_D'])
    parser.add_argument('--ckpt',  default='lora_best.pt')
    parser.add_argument('--fps',   type=int, default=10)
    args = parser.parse_args()

    RESULTS    = ENH_DIR / f'results_mcfm_{args.mode}_realgt_lora_seed6'
    CKPT_DIR   = RESULTS / 'lora_ckpts'
    MCFM_DIR   = ENH_DIR / f'results_mcfm_{args.mode}_seed6_fixednoise' / 'beta0p0'
    RENDER_DIR = RESULTS / 'renders_lora_best'
    SBS_DIR    = RESULTS / 'sbs_frames'
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    SBS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'=== render_lora_all_frames: mode={args.mode}  ckpt={args.ckpt} ===')

    ckpt_path = CKPT_DIR / args.ckpt
    assert ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}'

    # Load pipeline — immediately offload heavy models before GPU-intensive ops
    print('\n[LOAD] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # Offload slat models to CPU — too large to co-exist with struct sampling on 11GB
    flow_model.cpu()
    pipeline.models['slat_decoder_mesh'].cpu()
    gc.collect(); torch.cuda.empty_cache()

    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, n_blocks=24).to(DEVICE)
    alpha = nn.Parameter(torch.tensor(0.5, device=DEVICE))

    print(f'[CKPT] Loading {ckpt_path.name}...')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    lora_blocks.load_state_dict(ckpt['lora_state'])
    alpha.data.copy_(ckpt['alpha'].to(DEVICE))
    print(f'  epoch={ckpt["epoch"]}  alpha={alpha.item():.4f}  task_loss={ckpt["avg_task_loss"]:.5f}')

    # Sample voxel structure (only DINOv2 + struct models on GPU here)
    print(f'\n[STRUCT] Sampling structure (STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(VIDEO_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}')

    # Encode all frames with DINO
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model']
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i)
        if i % 50 == 0: print(f'  {i}/{N_FRAMES}')

    # Free all GPU models except what we need for rendering
    for name in list(pipeline.models.keys()):
        try: pipeline.models[name].cpu()
        except: pass
    gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    renderer = make_renderer()
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.to(DEVICE)
    flow_model.eval()

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    # Render all 150 frames
    print(f'\n[RENDER] Rendering {N_FRAMES} frames...')
    for fi in range(1, N_FRAMES + 1):
        out_path = RENDER_DIR / f'frame_{fi:04d}.png'
        if out_path.exists():
            continue
        color = render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                                    raw_tokens, args.mode, fi, coords,
                                    fixed_noise_feats, renderer)
        arr = (color.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_path)
        if fi % 10 == 0 or fi == N_FRAMES:
            print(f'  {fi}/{N_FRAMES}')

    # Build side-by-side frames: GT | MCFM | LoRA
    print('\n[SBS] Building comparison frames (GT | MCFM | LoRA)...')
    cell     = 320
    label_h  = 28
    labels   = ['GT video', f'MCFM ({args.mode})', f'LoRA e{ckpt["epoch"]:02d}']
    for fi in range(1, N_FRAMES + 1):
        sbs_path = SBS_DIR / f'frame_{fi:04d}.png'
        if sbs_path.exists(): continue
        panels = []
        for p in [VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png',
                  MCFM_DIR / f'frame_{fi:04d}.png',
                  RENDER_DIR / f'frame_{fi:04d}.png']:
            if p.exists():
                panels.append(Image.open(p).convert('RGB').resize((cell, cell), Image.LANCZOS))
            else:
                panels.append(Image.new('RGB', (cell, cell), (40, 40, 40)))

        canvas = Image.new('RGB', (cell * 3, cell + label_h), (20, 20, 20))
        draw   = ImageDraw.Draw(canvas)
        for col, (img, lbl) in enumerate(zip(panels, labels)):
            canvas.paste(img, (col * cell, label_h))
            draw.rectangle([col*cell, 0, (col+1)*cell-1, label_h-1], fill=(35, 35, 50))
            try:   tw = draw.textbbox((0,0), lbl, font=font)[2]
            except: tw = len(lbl) * 8
            draw.text((col*cell + (cell-tw)//2, 7), lbl, fill=(220,220,220), font=font)
        draw.text((cell*3 - 50, cell+label_h-16), f'f{fi:03d}', fill=(140,140,140), font=font)
        canvas.save(sbs_path)

    print('\n[VIDEO] Encoding...')
    out_mp4 = RESULTS / f'comparison_{args.mode}_lora_best.mp4'
    make_video(SBS_DIR, out_mp4, args.fps)

    print(f'\n=== DONE ===')
    print(f'  Renders : {RENDER_DIR}')
    print(f'  Video   : {out_mp4}')
    print(f'\nSCP:')
    print(f'  scp rajhansini@ranalab.cs.uchicago.edu:{out_mp4} ./')


if __name__ == '__main__':
    main()
