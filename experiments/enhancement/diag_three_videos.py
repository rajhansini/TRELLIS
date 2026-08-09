"""
Produce 3 comparison videos:
  video1_raw_perframe.mp4  : GT | Raw TRELLIS (per-frame noise, seed=frame_idx)
  video2_raw_fixed.mp4     : GT | Raw TRELLIS (fixed noise, seed=6)
  video3_mcfm_fixed.mp4    : GT | MCFM v2_C | MCFM v2_D | MCFM v3_C | MCFM v3_D

Only renders raw TRELLIS per-frame noise (150 frames).
All other materials already exist.
"""
import sys, os, gc, subprocess
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from step8_decode_render.decode_render import make_renderer, normalize_slat, decode_and_render

DEVICE      = torch.device('cuda')
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
N_FRAMES    = 150
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'

ENH_DIR          = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')
VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')

# already rendered
RAW_FIXED_DIR = ENH_DIR / 'diag_full_comparison' / 'raw_trellis'
MCFM_DIRS = {
    'v2_C': ENH_DIR / 'results_mcfm_v2_C_seed6_fixednoise' / 'beta0p0',
    'v2_D': ENH_DIR / 'results_mcfm_v2_D_seed6_fixednoise' / 'beta0p0',
    'v3_C': ENH_DIR / 'results_mcfm_v3_C_seed6_fixednoise' / 'beta0p0',
    'v3_D': ENH_DIR / 'results_mcfm_v3_D_seed6_fixednoise' / 'beta0p0',
}

OUT_DIR         = ENH_DIR / 'diag_three_videos'
RAW_PERFRAME_DIR = OUT_DIR / 'raw_trellis_perframe'
SBS1_DIR        = OUT_DIR / 'sbs1_perframe'
SBS2_DIR        = OUT_DIR / 'sbs2_fixed'
SBS3_DIR        = OUT_DIR / 'sbs3_mcfm'
for d in [OUT_DIR, RAW_PERFRAME_DIR, SBS1_DIR, SBS2_DIR, SBS3_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG = OUT_DIR / 'run.log'

_t_seq  = torch.linspace(1, 0, STEPS + 1).tolist()
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def encode_video(frames_dir, out_mp4, fps=10):
    fl = frames_dir / '_list.txt'
    with open(fl, 'w') as f:
        for p in sorted(frames_dir.glob('frame_*.png')):
            f.write(f"file '{p}'\nduration {1.0/fps:.6f}\n")
    ffmpeg = next((x for x in ['/usr/bin/ffmpeg', 'ffmpeg']
                   if subprocess.run(['which', x] if x == 'ffmpeg' else ['test', '-f', x],
                                     capture_output=True).returncode == 0), None)
    if ffmpeg is None:
        ffmpeg = os.environ.get('FFMPEG', 'ffmpeg')
    r = subprocess.run(
        [ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(fl),
         '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', '-b:v', '8M', str(out_mp4)],
        capture_output=True, text=True)
    fl.unlink(missing_ok=True)
    if r.returncode == 0:
        log(f'  saved: {out_mp4}  ({out_mp4.stat().st_size/1e6:.1f} MB)')
    else:
        log(f'  ffmpeg error: {r.stderr[-400:]}')


def make_sbs(panels_info, out_dir, fi, cell=320, label_h=28, font=None):
    out = out_dir / f'frame_{fi:04d}.png'
    if out.exists():
        return
    n = len(panels_info)
    canvas = Image.new('RGB', (cell * n, cell + label_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (path, lbl) in enumerate(panels_info):
        img = (Image.open(path).convert('RGB').resize((cell, cell), Image.LANCZOS)
               if path.exists() else Image.new('RGB', (cell, cell), (40, 40, 40)))
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col*cell, 0, (col+1)*cell-1, label_h-1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0,0), lbl, font=font)[2]
        except: tw = len(lbl) * 7
        draw.text((col*cell + (cell-tw)//2, 6), lbl, fill=(210,210,210), font=font)
    draw.text((cell*n - 55, cell+label_h-15), f'f{fi:03d}', fill=(120,120,120), font=font)
    canvas.save(out)


# ── STEP 1: render raw TRELLIS per-frame noise ────────────────────────────────
already = sum(1 for i in range(1,N_FRAMES+1) if (RAW_PERFRAME_DIR/f'frame_{i:04d}.png').exists())
log(f'=== diag_three_videos ===')
log(f'[RAW-PERFRAME] {already}/{N_FRAMES} frames already done.')

if already < N_FRAMES:
    log('\n[LOAD] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    flow_model.eval()

    log(f'\n[STRUCT] Sampling structure (frame 75, STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(VIDEO_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    log(f'  N_vox={N_vox}')
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    log(f'\n[DINO] Pre-encoding all {N_FRAMES} frames...')
    cond_tensors = {}
    for i in range(1, N_FRAMES + 1):
        img = Image.open(VIDEO_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB')
        with torch.no_grad():
            cond_tensors[i] = pipeline.encode_image([img]).cpu()
        if i % 50 == 0:
            log(f'  {i}/{N_FRAMES}')

    renderer = make_renderer()

    log(f'\n[RENDER] Per-frame noise (seed=frame_idx)...')
    for fi in range(1, N_FRAMES + 1):
        out_path = RAW_PERFRAME_DIR / f'frame_{fi:04d}.png'
        if out_path.exists():
            continue
        torch.manual_seed(fi)   # ← different noise per frame
        noise   = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
        cond_gl = cond_tensors[fi].to(DEVICE)
        ns = sp.SparseTensor(feats=noise, coords=coords)
        with torch.no_grad():
            for t, t_prev in T_PAIRS:
                t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                v  = flow_model(ns, t_ten, cond_gl)
                ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
        slat = normalize_slat(ns)
        del ns; gc.collect(); torch.cuda.empty_cache()
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        del slat; gc.collect(); torch.cuda.empty_cache()
        arr = (color.detach().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_path)
        if fi % 10 == 0 or fi == N_FRAMES:
            log(f'  {fi}/{N_FRAMES}')
    log('  Per-frame noise render done.')

# ── STEP 2: build SBS frames ──────────────────────────────────────────────────
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except:
    font = ImageFont.load_default()

log('\n[SBS1] GT | Raw per-frame noise...')
for fi in range(1, N_FRAMES + 1):
    make_sbs([
        (VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
        (RAW_PERFRAME_DIR / f'frame_{fi:04d}.png', 'Raw TRELLIS (per-frame noise)'),
    ], SBS1_DIR, fi, font=font)

log('[SBS2] GT | Raw fixed noise...')
for fi in range(1, N_FRAMES + 1):
    make_sbs([
        (VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
        (RAW_FIXED_DIR    / f'frame_{fi:04d}.png', 'Raw TRELLIS (fixed noise, seed=6)'),
    ], SBS2_DIR, fi, font=font)

log('[SBS3] GT | MCFM v2_C | v2_D | v3_C | v3_D (fixed noise)...')
for fi in range(1, N_FRAMES + 1):
    make_sbs([
        (VIDEO_FRAMES_DIR          / f'frame_{fi:04d}.png', 'GT video'),
        (MCFM_DIRS['v2_C']         / f'frame_{fi:04d}.png', 'MCFM v2_C (fixed noise)'),
        (MCFM_DIRS['v2_D']         / f'frame_{fi:04d}.png', 'MCFM v2_D (fixed noise)'),
        (MCFM_DIRS['v3_C']         / f'frame_{fi:04d}.png', 'MCFM v3_C (fixed noise)'),
        (MCFM_DIRS['v3_D']         / f'frame_{fi:04d}.png', 'MCFM v3_D (fixed noise)'),
    ], SBS3_DIR, fi, font=font)

# ── STEP 3: encode videos ─────────────────────────────────────────────────────
log('\n[VIDEO] Encoding 3 videos...')
encode_video(SBS1_DIR, OUT_DIR / 'video1_raw_perframe.mp4')
encode_video(SBS2_DIR, OUT_DIR / 'video2_raw_fixed.mp4')
encode_video(SBS3_DIR, OUT_DIR / 'video3_mcfm_fixed.mp4')

log('\n=== DONE ===')
log(f'  Output: {OUT_DIR}')
log('\nSCP all 3:')
for v in ['video1_raw_perframe.mp4', 'video2_raw_fixed.mp4', 'video3_mcfm_fixed.mp4']:
    log(f'  scp rajhansini@fe01.ai.cs.uchicago.edu:{OUT_DIR}/{v} ~/Downloads/')
