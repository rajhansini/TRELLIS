"""
build_slat_cache_obj.py
-----------------------
Frozen per-frame SLaT cache for an ARBITRARY object, in the exact format
solve_mesh_alignment.py expects.

WHY THIS EXISTS
  The alignment solver reads RUN_DIR/slat_cache.npz, which until now only ever
  appeared as a byproduct of the decoder-side rungs (rung5, rung13) and only for
  the teapot. Running the KL sweep on spot_lava and teapot_lava2 needs the same
  cache for those objects, and nothing produced it.

FORMAT, matched to the existing teapot cache byte for byte:
      slats   [n_frames, N_vox, 8]  float32   POST normalize_slat, i.e. already
                                              in mesh-decoder space, because the
                                              solver feeds it straight to
                                              slat_decoder_mesh with no further
                                              scaling.
      coords  [N_vox, 4]            int32     leading batch column included.

N_VOX IS NOT 7301 IN GENERAL.  That number is the teapot's, from
sample_sparse_structure on its reference frame.  Every object gets its own, so
this script RECORDS it rather than asserting it, and writes it into meta.  Any
downstream script that hard-codes 7301 will be wrong for these objects.

DETERMINISM.  Structure seed and noise seed are the same constants the rungs use
(STRUCT_SEED=42, NOISE_SEED=6) and the structure is sampled ONCE from the
reference frame, then shared by all frames -- identical to what the teapot runs
do.  Only the DINO conditioning varies per frame.

Usage:
  python build_slat_cache_obj.py \
      --frames-dir /net/.../TRELLIS.2/data/spot_lava/frames_from_video \
      --out        .../alignment_spot_lava/slat_cache.npz
"""

import sys, os, argparse, json, gc
from pathlib import Path

os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--frames-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ref-frame', type=int, default=75,
                help='frame the sparse structure is sampled from. 75 matches '
                     'the teapot runs.')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--steps', type=int, default=25)
args = ap.parse_args()

OUT = Path(args.out); OUT.parent.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from step8_decode_render.decode_render import normalize_slat

DEVICE = torch.device('cuda')
PRETRAINED, STRUCT_SEED, NOISE_SEED = 'JeffreyXiang/TRELLIS-image-large', 42, 6
RESCALE_T = 3.0
_ts = np.linspace(1, 0, args.steps + 1)
_ts = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(args.steps)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
GT = Path(args.frames_dir)


def encode(dino, i):
    p = GT / f'frame_{i:04d}.png'
    assert p.exists(), f'missing frame: {p}'
    img = Image.open(p).convert('RGB').resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


def main():
    print('=' * 84)
    print(f'SLAT CACHE  frames={GT}  ref={args.ref_frame}  n={args.n_frames}')
    print('=' * 84, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED); pipe.to(DEVICE)
    fm = pipe.models['slat_flow_model']
    for p in fm.parameters():
        p.requires_grad_(False)

    ref = Image.open(GT / f'frame_{args.ref_frame:04d}.png').convert('RGB')
    cs = pipe.get_cond([ref])
    torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    n_vox = int(coords.shape[0])
    print(f'[STRUCT] N_vox = {n_vox:,}  (teapot is 7,301; this object has its own)',
          flush=True)
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass

    dino = pipe.models['image_cond_model'].to(DEVICE)
    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(n_vox, fm.in_channels, device=DEVICE)

    slats = np.zeros((args.n_frames, n_vox, 8), np.float32)
    import time
    t0 = time.time()
    for i in range(1, args.n_frames + 1):
        cond = encode(dino, i).unsqueeze(0).to(DEVICE)
        x = sp.SparseTensor(feats=noise.clone(), coords=coords)
        with torch.no_grad():
            for t, tp in T_PAIRS:
                tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
                x = x.replace(x.feats - (t - tp) * fm(x, tt, cond).feats)
            slats[i - 1] = normalize_slat(x).feats.float().cpu().numpy()
        del x, cond
        if i % 10 == 0 or i == args.n_frames:
            el = time.time() - t0
            print(f'  {i:3d}/{args.n_frames}  {el:5.0f}s  '
                  f'eta {el / i * (args.n_frames - i):5.0f}s', flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # A latent that is all-zero or non-finite means the flow silently failed for
    # that frame and every mesh built from it would be garbage.
    bad = [i + 1 for i in range(args.n_frames)
           if not np.isfinite(slats[i]).all() or np.abs(slats[i]).max() < 1e-6]
    assert not bad, f'degenerate SLaT on frames {bad[:10]}'

    np.savez_compressed(OUT, slats=slats,
                        coords=coords.cpu().numpy().astype(np.int32),
                        meta=json.dumps(dict(frames_dir=str(GT), n_vox=n_vox,
                                             ref_frame=args.ref_frame,
                                             n_frames=args.n_frames,
                                             steps=args.steps,
                                             struct_seed=STRUCT_SEED,
                                             noise_seed=NOISE_SEED)))
    print(f'\n[DONE] {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)')
    print(f'  slats {slats.shape}  coords {tuple(coords.shape)}  N_vox {n_vox:,}')
    print(f'  |slat| mean {np.abs(slats).mean():.4f}  max {np.abs(slats).max():.4f}')


if __name__ == '__main__':
    main()
