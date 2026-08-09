"""
save_slat_cache.py
------------------
Generates and saves the SLaT cache for all rung5 runs using the EXACT same
pipeline as rung5_subladder.py's precompute_slats — correct DINOv2 encoding,
same seeds (STRUCT_SEED=42, NOISE_SEED=6).

MUST run on an L40S or A40 GPU — RTX 2080 Ti gives N_vox=7253 instead of 7301.
All rung5 runs use the same seeds, so one cache serves all 10 runs.

Usage (run on Slurm L40S/A40 node):
  python save_slat_cache.py
  python save_slat_cache.py --out-path /path/to/rung5_slat_cache.npz
"""
import sys, os, gc, argparse
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
                help='Generate only this single frame index (1-150). Faster sanity check.')
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

# ── DINOv2 encode all frames — IDENTICAL to rung5_subladder.py ────────────────
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

# ── sparse structure ───────────────────────────────────────────────────────────
print('[STRUCT] Sampling structure...')
struct_img  = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([pipeline.preprocess_image(struct_img)])
dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

torch.manual_seed(STRUCT_SEED)
with torch.no_grad():
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1).to(DEVICE)
import socket
N_vox = coords.shape[0]
print(f'[STRUCT] N_vox={N_vox}  node={socket.gethostname()}')
assert N_vox == 7301, (
    f'N_vox={N_vox} != 7301 — wrong GPU type. Training used L40S or A40. '
    f'RTX 2080 Ti gives 7253. Run this script on an L40S/A40 Slurm node.'
)

torch.manual_seed(NOISE_SEED)
fixed_noise = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

# ── precompute all 150 SLaTs — IDENTICAL to rung5_subladder.py ────────────────
def full_denoise(cond_gl):
    ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v  = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)

print(f'[SLAT] Precomputing {len(FRAME_LIST)} SLaT(s)...')
slat_feats = {}   # fi → np array (N_vox, 8)
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

# Save as full (150, N_vox, 8) array; untouched frames are zeros.
slats_out = np.zeros((N_FRAMES, N_vox, 8), dtype=np.float32)
for fi, arr in slat_feats.items():
    slats_out[fi - 1] = arr

print('[SAVE] Writing slat_cache.npz...')
np.savez(OUT_PATH, slats=slats_out, coords=slat_coords_save)
print(f'[DONE] {OUT_PATH}  frames={list(slat_feats.keys())}  N_vox={N_vox}')
