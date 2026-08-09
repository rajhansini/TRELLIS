"""
Build 6-panel comparison video:
  GT | Raw TRELLIS | MCFM v2_C | MCFM v2_D | MCFM v3_C | MCFM v3_D
  (no LoRA anywhere)

Only renders raw TRELLIS (150 frames) — the 4 MCFM blended dirs already exist.
Run on h002 (H200).
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

# existing blended dirs
MCFM_DIRS = {
    'v2_C': ENH_DIR / 'results_mcfm_v2_C_seed6_fixednoise' / 'beta0p0',
    'v2_D': ENH_DIR / 'results_mcfm_v2_D_seed6_fixednoise' / 'beta0p0',
    'v3_C': ENH_DIR / 'results_mcfm_v3_C_seed6_fixednoise' / 'beta0p0',
    'v3_D': ENH_DIR / 'results_mcfm_v3_D_seed6_fixednoise' / 'beta0p0',
}

OUT_DIR  = ENH_DIR / 'diag_full_comparison'
RAW_DIR  = OUT_DIR / 'raw_trellis_perframe_noise'
SBS_DIR  = OUT_DIR / 'sbs_frames'
for d in [OUT_DIR, RAW_DIR, SBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG = OUT_DIR / 'run.log'

_t_seq  = torch.linspace(1, 0, STEPS + 1).tolist()
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def make_video(frames_dir, out_mp4, fps=10):
    fl = frames_dir / '_list.txt'
    with open(fl, 'w') as f:
        for p in sorted(frames_dir.glob('frame_*.png')):
            f.write(f"file '{p}'\nduration {1.0/fps:.6f}\n")
    for ffmpeg in ['/usr/bin/ffmpeg', 'ffmpeg']:
        r = subprocess.run(
            [ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(fl),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out_mp4)],
            capture_output=True, text=True)
        if r.returncode == 0:
            log(f'  {out_mp4}  ({out_mp4.stat().st_size/1e6:.1f} MB)')
            fl.unlink(missing_ok=True)
            return
    log(f'  ffmpeg error: {r.stderr[-300:]}')
    fl.unlink(missing_ok=True)


# ── STEP 1: render raw TRELLIS 150 frames ─────────────────────────────────────
already_done = sum(1 for i in range(1, N_FRAMES+1) if (RAW_DIR/f'frame_{i:04d}.png').exists())
if already_done == N_FRAMES:
    log(f'[RAW] All {N_FRAMES} raw TRELLIS frames already exist, skipping render.')
else:
    log('=== diag_full_comparison ===')
    log(f'\n[LOAD] Loading pipeline...')
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
            cond_tensors[i] = pipeline.encode_image([img]).cpu()  # (1, 1374, 1024)
        if i % 50 == 0:
            log(f'  {i}/{N_FRAMES}')

    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    renderer = make_renderer()

    log(f'\n[RAW] Rendering {N_FRAMES} raw TRELLIS frames...')
    for fi in range(1, N_FRAMES + 1):
        out_path = RAW_DIR / f'frame_{fi:04d}.png'
        if out_path.exists():
            continue
        cond_gl = cond_tensors[fi].to(DEVICE)
        ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
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

    log('  Raw TRELLIS render complete.')


# ── STEP 2: build 6-panel SBS frames ──────────────────────────────────────────
log('\n[SBS] Building 6-panel frames...')
cell    = 256
label_h = 26
labels  = ['GT video', 'Raw TRELLIS', 'MCFM v2_C', 'MCFM v2_D', 'MCFM v3_C', 'MCFM v3_D']
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
except:
    font = ImageFont.load_default()

for fi in range(1, N_FRAMES + 1):
    sbs_path = SBS_DIR / f'frame_{fi:04d}.png'
    if sbs_path.exists():
        continue
    src_paths = [
        VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png',
        RAW_DIR          / f'frame_{fi:04d}.png',
        MCFM_DIRS['v2_C'] / f'frame_{fi:04d}.png',
        MCFM_DIRS['v2_D'] / f'frame_{fi:04d}.png',
        MCFM_DIRS['v3_C'] / f'frame_{fi:04d}.png',
        MCFM_DIRS['v3_D'] / f'frame_{fi:04d}.png',
    ]
    panels = [Image.open(p).convert('RGB').resize((cell, cell), Image.LANCZOS)
              if p.exists() else Image.new('RGB', (cell, cell), (40,40,40))
              for p in src_paths]

    canvas = Image.new('RGB', (cell * 6, cell + label_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (img, lbl) in enumerate(zip(panels, labels)):
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col*cell, 0, (col+1)*cell-1, label_h-1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0,0), lbl, font=font)[2]
        except: tw = len(lbl) * 7
        draw.text((col*cell + (cell-tw)//2, 6), lbl, fill=(210,210,210), font=font)
    draw.text((cell*6 - 50, cell + label_h - 14), f'f{fi:03d}', fill=(120,120,120), font=font)
    canvas.save(sbs_path)

log('  SBS frames done.')

# ── STEP 3: encode video ───────────────────────────────────────────────────────
log('\n[VIDEO] Encoding...')
out_mp4 = OUT_DIR / 'gt_raw_v2C_v2D_v3C_v3D.mp4'
make_video(SBS_DIR, out_mp4, fps=10)

log('\n=== DONE ===')
log(f'  Video: {out_mp4}')
log('\nSCP:')
log(f'  scp rajhansini@fe01.ai.cs.uchicago.edu:{out_mp4} ~/Downloads/')
