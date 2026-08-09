"""
Render all 150 frames for:
  1. Raw TRELLIS  — no blending, no LoRA
  2. MCFM v3_C   — blended conditioning, no LoRA
Then build 3-panel video: GT | Raw TRELLIS | MCFM v3_C

Run on h002 (H200, 143 GB) — no CPU offloading needed.
"""
import sys, os, gc, subprocess
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from step4_mcfm.mcfm import mcfm_v3
from step8_decode_render.decode_render import make_renderer, normalize_slat, decode_and_render

DEVICE      = torch.device('cuda')
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
N_FRAMES    = 150
MODE        = 'v3_C'
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'

VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_raw_vs_mcfm')
RAW_DIR  = OUT_DIR / 'raw_trellis'
MCFM_DIR = OUT_DIR / 'mcfm_v3C'
SBS_DIR  = OUT_DIR / 'sbs_frames'
for d in [OUT_DIR, RAW_DIR, MCFM_DIR, SBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG = OUT_DIR / 'run.log'

_t_seq  = torch.linspace(1, 0, STEPS + 1).tolist()
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def get_window(frame_idx):
    return [frame_idx, min(N_FRAMES, frame_idx + 1)]


def blend_v3c(raw_tokens, frame_idx):
    win    = get_window(frame_idx)
    lam    = torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)
    tok_d  = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    K_hat, _ = mcfm_v3(tok_d, win, frame_idx, lam)
    return K_hat   # (1374, 1024)


def encode_frame(pipeline, frame_idx):
    img  = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    with torch.no_grad():
        tok = pipeline.encode_image([img])   # (1, 1374, 1024)
    return tok.squeeze(0).cpu()              # (1374, 1024)


def run_flow(flow_model, ns, cond_gl):
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v  = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return ns


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
            log(f'  Video: {out_mp4}  ({out_mp4.stat().st_size/1e6:.1f} MB)')
            fl.unlink(missing_ok=True)
            return
        if 'libx264' not in r.stderr and 'No such file' not in r.stderr:
            break
    log(f'  ffmpeg error: {r.stderr[-300:]}')
    fl.unlink(missing_ok=True)


# ── Load pipeline ──────────────────────────────────────────────────────────────
log('=== diag_raw_vs_mcfm ===')
log(f'  MODE={MODE}  STRUCT_SEED={STRUCT_SEED}  FIXED_SEED={FIXED_SEED}')

log('\n[LOAD] Loading pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)
flow_model = pipeline.models['slat_flow_model']
flow_model.eval()

# ── Sample structure ───────────────────────────────────────────────────────────
log(f'\n[STRUCT] Sampling structure (frame 75, STRUCT_SEED={STRUCT_SEED})...')
img_75      = Image.open(VIDEO_FRAMES_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([img_75])
torch.manual_seed(STRUCT_SEED)
coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
N_vox  = coords.shape[0]
log(f'  N_vox={N_vox}')
del cond_struct; gc.collect(); torch.cuda.empty_cache()

# ── Pre-encode all 150 frames ─────────────────────────────────────────────────
log(f'\n[DINO] Pre-encoding all {N_FRAMES} frames...')
raw_tokens = {}
for i in range(1, N_FRAMES + 1):
    raw_tokens[i] = encode_frame(pipeline, i)
    if i % 30 == 0:
        log(f'  {i}/{N_FRAMES}')
log(f'  Done. Token shape: {raw_tokens[1].shape}')

# ── Fixed noise ────────────────────────────────────────────────────────────────
torch.manual_seed(FIXED_SEED)
fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
log(f'\n  fixed_noise_feats: {fixed_noise_feats.shape}')

renderer = make_renderer()

# ── Render loop ────────────────────────────────────────────────────────────────
log(f'\n[RENDER] Rendering {N_FRAMES} frames (raw + MCFM)...')
for fi in range(1, N_FRAMES + 1):
    raw_path  = RAW_DIR  / f'frame_{fi:04d}.png'
    mcfm_path = MCFM_DIR / f'frame_{fi:04d}.png'

    need_raw  = not raw_path.exists()
    need_mcfm = not mcfm_path.exists()
    if not need_raw and not need_mcfm:
        continue

    # ── Raw TRELLIS ──
    if need_raw:
        cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)   # (1, 1374, 1024)
        ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        ns = run_flow(flow_model, ns, cond_gl)
        slat = normalize_slat(ns)
        del ns; gc.collect(); torch.cuda.empty_cache()
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        del slat; gc.collect(); torch.cuda.empty_cache()
        arr = (color.detach().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(raw_path)

    # ── MCFM v3_C ──
    if need_mcfm:
        K_hat   = blend_v3c(raw_tokens, fi)
        cond_gl = K_hat.unsqueeze(0)                        # (1, 1374, 1024)
        ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        ns = run_flow(flow_model, ns, cond_gl)
        slat = normalize_slat(ns)
        del ns, K_hat; gc.collect(); torch.cuda.empty_cache()
        color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
        del slat; gc.collect(); torch.cuda.empty_cache()
        arr = (color.detach().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(mcfm_path)

    if fi % 10 == 0 or fi == N_FRAMES:
        log(f'  {fi}/{N_FRAMES}')

# ── Build 3-panel side-by-side frames ─────────────────────────────────────────
log('\n[SBS] Building 3-panel frames (GT | Raw TRELLIS | MCFM v3_C)...')
cell    = 320
label_h = 28
labels  = ['GT video', 'Raw TRELLIS (no blend)', 'MCFM v3_C (no LoRA)']
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except:
    font = ImageFont.load_default()

for fi in range(1, N_FRAMES + 1):
    sbs_path = SBS_DIR / f'frame_{fi:04d}.png'
    if sbs_path.exists():
        continue
    paths = [
        VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png',
        RAW_DIR          / f'frame_{fi:04d}.png',
        MCFM_DIR         / f'frame_{fi:04d}.png',
    ]
    panels = [Image.open(p).convert('RGB').resize((cell, cell), Image.LANCZOS)
              if p.exists() else Image.new('RGB', (cell, cell), (40,40,40))
              for p in paths]

    canvas = Image.new('RGB', (cell * 3, cell + label_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    for col, (img, lbl) in enumerate(zip(panels, labels)):
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col*cell, 0, (col+1)*cell-1, label_h-1], fill=(35, 35, 50))
        try:   tw = draw.textbbox((0,0), lbl, font=font)[2]
        except: tw = len(lbl) * 8
        draw.text((col*cell + (cell-tw)//2, 6), lbl, fill=(220,220,220), font=font)
    draw.text((cell*3 - 55, cell + label_h - 16), f'f{fi:03d}', fill=(130,130,130), font=font)
    canvas.save(sbs_path)

# ── Encode video ───────────────────────────────────────────────────────────────
log('\n[VIDEO] Encoding...')
out_mp4 = OUT_DIR / 'gt_vs_raw_vs_mcfm_v3C.mp4'
make_video(SBS_DIR, out_mp4, fps=10)

log('\n=== DONE ===')
log(f'  Video: {out_mp4}')
log('\nSCP:')
log(f'  scp rajhansini@fe01.ai.cs.uchicago.edu:{out_mp4} ~/Downloads/')
