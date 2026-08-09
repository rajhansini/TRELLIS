"""
debug_save_slat_cache.py
------------------------
Matches rung5_subladder.py's EXACT operation order:
  1. Load pipeline (all models on GPU)
  2. Structure sampling FIRST — before any DINOv2 work, raw PIL image,
     no preprocess_image wrapper — identical to training lines 839-843
  3. Offload non-essential pipeline models
  4. DINOv2 encode frames
  5. Flow model denoise all SLaTs

Root cause of N_vox variation: cuDNN benchmarks conv algorithms on first use.
Running DINOv2 encoding BEFORE structure sampling (as original save_slat_cache.py
did) pollutes the cuDNN algorithm cache → different algorithm selected for the
sparse structure decoder → different FP results → different N_vox.
"""
import sys, os, gc, argparse, socket
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))
os.environ['HF_HOME']         = '/net/scratch/rajhansini/.cache/huggingface'
os.environ.setdefault('SPCONV_ALGO',  'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')

GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
PIPELINE_CKPT = 'JeffreyXiang/TRELLIS-image-large'
STRUCT_SEED   = 42
NOISE_SEED    = 6
N_FRAMES      = 150
DEVICE        = torch.device('cuda')
STEPS         = 25
RESCALE_T     = 3.0

_t_seq = np.linspace(1, 0, STEPS + 1)
_t_seq = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

SHARED_CACHE = ROOT / 'experiments' / 'decoder_same_geometry' / 'rung5_slat_cache.npz'

ap = argparse.ArgumentParser()
ap.add_argument('--out-path', type=Path, default=SHARED_CACHE)
ap.add_argument('--frame', type=int, default=None,
                help='Single frame index (1-150) for quick sanity check.')
args = ap.parse_args()
OUT_PATH   = args.out_path.resolve()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
FRAME_LIST = [args.frame] if args.frame is not None else list(range(1, N_FRAMES + 1))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import normalize_slat
import torchvision.transforms as T

print('[LOAD] pipeline...')
pipeline   = TrellisImageTo3DPipeline.from_pretrained(PIPELINE_CKPT)
pipeline.cuda()
flow_model = pipeline.models['slat_flow_model']
dino_model = pipeline.models['image_cond_model']
for m in [flow_model, dino_model]:
    for p in m.parameters(): p.requires_grad_(False)

# ── STEP 1: structure sampling — BEFORE any DINOv2 work, matching training ────
# Training (rung5_subladder.py line 841): pipeline.get_cond([ref_img])
# Raw PIL image, no preprocess_image wrapper. All models on GPU.
print(f'[STRUCT] seed={STRUCT_SEED}  node={socket.gethostname()}')
ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([ref_img])
torch.manual_seed(STRUCT_SEED)
with torch.no_grad():
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1).to(DEVICE)
N_vox = coords.shape[0]
print(f'[STRUCT] N_vox={N_vox}  node={socket.gethostname()}')
del cond_struct; gc.collect(); torch.cuda.empty_cache()

# Offload non-essential pipeline models (matching training lines 850-853)
for name in list(pipeline.models.keys()):
    if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
        try: pipeline.models[name].cpu()
        except Exception: pass
torch.cuda.empty_cache()

torch.manual_seed(NOISE_SEED)
fixed_noise = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

# ── STEP 2: DINOv2 encode frames (after structure sampling) ───────────────────
_dino_tfm = T.Compose([
    T.Resize((518, 518), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def encode_frame(fi):
    img = Image.open(GT_FRAMES_DIR / f'frame_{fi:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    t   = _dino_tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(t, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0).cpu()

print(f'[DINO] Encoding {len(FRAME_LIST)} frame(s)...')
raw_tokens = {}
for fi in FRAME_LIST:
    raw_tokens[fi] = encode_frame(fi)
    print(f'  frame {fi} done')

# ── STEP 3: flow model denoise all SLaTs ──────────────────────────────────────
def full_denoise(cond_gl):
    ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v  = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)

print(f'[SLAT] Precomputing {len(FRAME_LIST)} SLaT(s)...')
slat_feats = {}
slat_coords_save = None
for fi in FRAME_LIST:
    cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
    slat = full_denoise(cond)
    slat_feats[fi] = slat.feats.cpu().numpy()
    if slat_coords_save is None:
        slat_coords_save = slat.coords.cpu().numpy()
    del slat, cond
    print(f'  frame {fi} done')
    gc.collect(); torch.cuda.empty_cache()

slats_out = np.zeros((N_FRAMES, N_vox, 8), dtype=np.float32)
for fi, arr in slat_feats.items():
    slats_out[fi - 1] = arr

print('[SAVE] Writing slat_cache.npz...')
np.savez(OUT_PATH, slats=slats_out, coords=slat_coords_save)
print(f'[DONE] {OUT_PATH}  frames={list(slat_feats.keys())}  N_vox={N_vox}')
