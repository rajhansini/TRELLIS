"""
Phase C and D experiments — no LoRA, no enhancement matrix.

Phase C: window = [i, i+1]  (current + next)
  C1: lc=0.99  ln=0.01
  C2: lc=0.90  ln=0.10
  C3: lc=0.50  ln=0.50
  C4: lc=0.10  ln=0.90

Phase D: window = [i-1, i, i+1]  (prev + current + next)
  D1: lp=0.10  lc=0.80  ln=0.10
  D2: lp=0.25  lc=0.50  ln=0.25
  D3: lp=0.33  lc=0.33  ln=0.33

Usage:
  python phase_cd.py --phase C --start_frame 77 --frames 1
  python phase_cd.py --phase D --start_frame 77 --frames 1
  python phase_cd.py --phase CD --start_frame 1 --frames 150
"""

import os, sys, argparse, math, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
from PIL import Image, ImageDraw
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent
_ROOT = _PIPE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp
from step1_input_prep.input_prep       import load_frame, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino, encode_frames

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = str(_PIPE / '..' / '..' / '..' /
                  'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
RESULTS_DIR = _PIPE / 'results' / 'phase_cd'
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

# ── Experiment definitions ────────────────────────────────────────────────────
PHASE_C = [
    dict(tag='C0',  lp=0.00, lc=1.00, ln=0.00),
    dict(tag='C1',  lp=0.00, lc=0.99, ln=0.01),
    dict(tag='C2',  lp=0.00, lc=0.95, ln=0.05),
    dict(tag='C3',  lp=0.00, lc=0.90, ln=0.10),
    dict(tag='C4',  lp=0.00, lc=0.80, ln=0.20),
    dict(tag='C5',  lp=0.00, lc=0.70, ln=0.30),
    dict(tag='C6',  lp=0.00, lc=0.60, ln=0.40),
    dict(tag='C7',  lp=0.00, lc=0.50, ln=0.50),
    dict(tag='C8',  lp=0.00, lc=0.30, ln=0.70),
    dict(tag='C9',  lp=0.00, lc=0.10, ln=0.90),
    dict(tag='C10', lp=0.00, lc=0.00, ln=1.00),
]
PHASE_D = [
    dict(tag='D1', lp=0.10, lc=0.80, ln=0.10),
    dict(tag='D2', lp=0.25, lc=0.50, ln=0.25),
    dict(tag='D3', lp=0.33, lc=0.33, ln=0.33),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def denoise(flow_model, noise_sp, cond_gl):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def render_slat(pipeline, slat, renderer):
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    mask  = result['mask'].unsqueeze(0)
    color = result['color'] * mask + (1.0 - mask)
    return (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def blend(all_tokens, lp, lc, ln):
    """Weighted blend of [prev, curr, next] tokens. lp=0 → C-style [curr, next]."""
    out = lc * all_tokens['curr']
    if ln > 0:
        out = out + ln * all_tokens['next']
    if lp > 0:
        out = out + lp * all_tokens['prev']
    return out


def make_comparison(gt_rgb, renders: dict, frame_idx: int, out_path: Path):
    """Side-by-side: GT | render per experiment."""
    W, H      = RENDER_RES, RENDER_RES
    pad       = 8
    label_h   = 24
    n         = 1 + len(renders)   # GT + experiments
    canvas_w  = n * W + (n + 1) * pad
    canvas_h  = H + 2 * pad + label_h
    canvas    = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
    draw      = ImageDraw.Draw(canvas)

    # GT
    gt = Image.fromarray(gt_rgb).resize((W, H), Image.LANCZOS)
    canvas.paste(gt, (pad, pad + label_h))
    draw.text((pad + W // 2 - 15, 6), 'GT', fill=(255, 255, 255))

    for i, (tag, img_arr) in enumerate(renders.items()):
        x = (i + 1) * (W + pad) + pad
        canvas.paste(Image.fromarray(img_arr), (x, pad + label_h))
        draw.text((x + W // 2 - 15, 6), tag, fill=(200, 200, 200))

    canvas.save(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='CD', choices=['C', 'D', 'CD'],
                        help='Which phase(s) to run')
    parser.add_argument('--tags', type=str, default=None,
                        help='Comma-separated experiment tags to run, e.g. C7,D2')
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames (default: all 150)')
    args = parser.parse_args()

    all_exps = []
    if 'C' in args.phase: all_exps += PHASE_C
    if 'D' in args.phase: all_exps += PHASE_D

    if args.tags:
        wanted = set(args.tags.split(','))
        experiments = [e for e in all_exps if e['tag'] in wanted]
    else:
        experiments = all_exps

    start    = args.start_frame
    n_frames = args.frames or N_FRAMES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'=== Phase {args.phase} ===')
    for e in experiments:
        print(f"  {e['tag']}: lp={e['lp']}  lc={e['lc']}  ln={e['ln']}")
    print(f'  frames {start}..{start+n_frames-1}  steps={STEPS}')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\nLoading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # Sample voxel structure first (needs image_cond_model on GPU)
    print('Sampling voxel structure from frame 75...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox: {N_vox}')

    # Offload non-essential models
    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── Pre-encode all frames ─────────────────────────────────────────────────
    print(f'\nPre-encoding {N_FRAMES} frames...')
    dino = load_dino(DEVICE)
    t0   = time.time()
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    print(f'  Done in {time.time()-t0:.1f}s')
    del dino
    torch.cuda.empty_cache()

    renderer   = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )
    flow_model.eval()

    # ── Per-frame loop ────────────────────────────────────────────────────────
    print(f'\nRendering {n_frames} frame(s) × {len(experiments)} experiments...')

    for frame_i in range(start, start + n_frames):
        frame_prev = max(1, frame_i - 1)
        frame_next = min(N_FRAMES, frame_i + 1)

        tok_prev = all_tokens[frame_prev]['tokens'].to(DEVICE)
        tok_curr = all_tokens[frame_i    ]['tokens'].to(DEVICE)
        tok_next = all_tokens[frame_next ]['tokens'].to(DEVICE)
        toks     = {'prev': tok_prev, 'curr': tok_curr, 'next': tok_next}

        gt_path  = GT_FRAMES_DIR / f'frame_{frame_i:04d}.png'
        gt_rgb   = np.array(Image.open(gt_path).convert('RGB').resize(
                       (RENDER_RES, RENDER_RES), Image.LANCZOS))

        renders = {}
        for exp in experiments:
            tag = exp['tag']
            K_hat   = blend(toks, exp['lp'], exp['lc'], exp['ln'])
            cond_gl = K_hat.unsqueeze(0)

            torch.manual_seed(NOISE_SEED + frame_i)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )

            x0   = denoise(flow_model, noise_sp, cond_gl)
            slat = normalize_slat(x0)
            del noise_sp, x0

            rendered = render_slat(pipeline, slat, renderer)
            del slat
            torch.cuda.empty_cache()

            # Save individual render
            exp_dir = RESULTS_DIR / tag
            exp_dir.mkdir(exist_ok=True)
            Image.fromarray(rendered).save(exp_dir / f'frame_{frame_i:04d}.png')
            renders[tag] = rendered

        # Save comparison grid: GT | C1 | C2 | ... | D1 | D2 | ...
        comp_path = RESULTS_DIR / f'comparison_{frame_i:04d}.png'
        make_comparison(gt_rgb, renders, frame_i, comp_path)
        print(f'  [{frame_i:03d}] saved comparison -> {comp_path.name}')

        del tok_prev, tok_curr, tok_next

    # ── Assemble per-experiment videos ────────────────────────────────────────
    if n_frames > 1:
        for exp in experiments:
            tag       = exp['tag']
            exp_dir   = RESULTS_DIR / tag
            vid_path  = exp_dir / 'render.mp4'
            subprocess.run([
                '/usr/bin/ffmpeg', '-y', '-framerate', '25',
                '-i', str(exp_dir / 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid_path)
            ], check=True)
            print(f'  Video: {vid_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
