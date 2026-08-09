"""
Diagnostic: raw TRELLIS on single frame (no blending, no LoRA).
Run frames 1, 75, 150 through vanilla TRELLIS and compare to real GT.
Tells us: does base TRELLIS conditioning reproduce lava texture at all?
"""
import sys, os, gc
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
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'

VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/diag_base')
OUT_DIR.mkdir(parents=True, exist_ok=True)

_t_seq  = torch.linspace(1, 0, STEPS + 1).tolist()
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]

TEST_FRAMES = [1, 75, 150]

LOG = OUT_DIR / 'diag_base_render.log'

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

log('=== diag_base_render: raw TRELLIS, no blend, no LoRA ===')
log(f'  OUT_DIR={OUT_DIR}')

# ── Load pipeline, offload heavy models ────────────────────────────────────────
log('\n[LOAD] Loading pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)
flow_model = pipeline.models['slat_flow_model']

flow_model.cpu()
pipeline.models['slat_decoder_mesh'].cpu()
gc.collect(); torch.cuda.empty_cache()

# ── Sample structure using frame 75 (same seed as training) ───────────────────
log(f'\n[STRUCT] Sampling structure (STRUCT_SEED={STRUCT_SEED}, frame 75)...')
img_75      = Image.open(VIDEO_FRAMES_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([img_75])
torch.manual_seed(STRUCT_SEED)
coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
N_vox  = coords.shape[0]
log(f'  N_vox={N_vox}')
del cond_struct
gc.collect(); torch.cuda.empty_cache()

# ── Pre-encode test frames with image_cond_model ──────────────────────────────
log(f'\n[DINO] Pre-encoding {TEST_FRAMES}...')
cond_tensors = {}
for fi in TEST_FRAMES:
    img  = Image.open(VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png').convert('RGB')
    cond = pipeline.get_cond([img])
    # cond['cond'] shape: (1, N_tokens, 1024), on GPU
    cond_tensors[fi] = cond['cond'].cpu()
    log(f'  frame {fi}: cond shape={cond_tensors[fi].shape}')
    del cond
    gc.collect(); torch.cuda.empty_cache()

# Offload image_cond_model — no longer needed
pipeline.models['image_cond_model'].cpu()
# Offload struct models
for name in pipeline.models:
    if name not in {'slat_flow_model', 'slat_decoder_mesh'}:
        try: pipeline.models[name].cpu()
        except: pass
gc.collect(); torch.cuda.empty_cache()

# ── Fixed noise (same as training/LoRA scripts) ───────────────────────────────
torch.manual_seed(FIXED_SEED)
fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels)
log(f'\n  fixed_noise_feats shape: {fixed_noise_feats.shape}  (FIXED_SEED={FIXED_SEED})')

renderer = make_renderer()
flow_model.eval()

# ── Render each test frame ─────────────────────────────────────────────────────
log(f'\n[RENDER] Rendering frames {TEST_FRAMES}...')
renders = {}
for fi in TEST_FRAMES:
    log(f'  frame {fi}...')
    cond_gl   = cond_tensors[fi].to(DEVICE)   # (1, N_tokens, 1024)
    noise     = fixed_noise_feats.clone().to(DEVICE)
    ns        = sp.SparseTensor(feats=noise, coords=coords)

    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()

    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v  = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)

    flow_model.cpu()
    del cond_gl, noise
    gc.collect(); torch.cuda.empty_cache()

    slat  = normalize_slat(ns)
    del ns
    gc.collect(); torch.cuda.empty_cache()

    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    pipeline.models['slat_decoder_mesh'].cpu()
    del slat
    gc.collect(); torch.cuda.empty_cache()

    renders[fi] = color.detach().clamp(0, 1).cpu()
    arr = (renders[fi].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_path = OUT_DIR / f'raw_trellis_f{fi:04d}.png'
    Image.fromarray(arr).save(out_path)
    log(f'    saved: {out_path}')

# ── GT | raw TRELLIS comparison strips ────────────────────────────────────────
log('\n[STRIP] Building comparison strips...')
cell = 320
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except:
    font = ImageFont.load_default()

label_h = 30
for fi in TEST_FRAMES:
    gt_img = Image.open(VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png').convert('RGB').resize((cell, cell), Image.LANCZOS)
    trellis_arr = (renders[fi].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    trellis_img = Image.fromarray(trellis_arr).resize((cell, cell), Image.LANCZOS)

    canvas = Image.new('RGB', (cell * 2, cell + label_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    for col, (img, lbl) in enumerate([(gt_img, f'GT  frame {fi}'),
                                       (trellis_img, f'Raw TRELLIS  f{fi}  (no blend, no LoRA)')]):
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col*cell, 0, (col+1)*cell - 1, label_h - 1], fill=(35, 35, 50))
        try:   tw = draw.textbbox((0,0), lbl, font=font)[2]
        except: tw = len(lbl) * 8
        draw.text((col*cell + (cell - tw)//2, 7), lbl, fill=(220, 220, 220), font=font)
    out = OUT_DIR / f'compare_f{fi:04d}.png'
    canvas.save(out)
    log(f'  saved: {out}')

log('\n=== DONE ===')
log(f'  Output: {OUT_DIR}')
log('\nSCP commands:')
for fi in TEST_FRAMES:
    log(f'  scp rajhansini@fe01.ai.cs.uchicago.edu:{OUT_DIR}/compare_f{fi:04d}.png ./')
