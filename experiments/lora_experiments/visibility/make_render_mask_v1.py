"""
make_render_mask_v1.py
----------------------
render_mask.npy for TRELLIS v1: the silhouette of OUR frozen mesh, from the
fixed training camera, for one object.

WHY
  TRELLIS.2's make_gt_targets.py builds a 2D target by taking the video's pixel
  at each pixel of OUR silhouette (no 3D round trip -- the camera is fixed, so
  screen location p in the video and in our render are the same point). It needs
  exactly one input beyond the video: render_mask.npy, our silhouette.

  That mask is model-specific. TRELLIS.2's was rendered from its own mesh; v1
  produces a different mesh, so v1 needs its own or every target is masked to
  the wrong shape.

WHAT IT DOES
  1. Samples the sparse structure from --ref-frame of THIS object.
  2. Runs the frozen 25-step flow, decodes, applies the alignment if one is
     given, renders the mask through the same MeshRenderer the loss uses.
  3. Writes render_mask.npy (bool [res, res]) plus a meta json.

ALIGNMENT IS OPTIONAL AND USUALLY WRONG HERE.  make_gt_targets.py measured that
a centroid+scale alignment made coverage WORSE (94.6% vs 97.8%), i.e. the video
is already correctly positioned and must not be warped. So --alignment defaults
to OFF; pass one only if you have solved it for this object and checked it.

Usage:
  python make_render_mask_v1.py \
      --frames-dir /net/.../TRELLIS.2/data/spot_lava/frames_from_video \
      --out .../gt_targets_v1_spot_lava/render_mask.npy
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
ap.add_argument('--ref-frame', type=int, default=75)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--alignment', default=None,
                help='alignment.json. OFF by default: make_gt_targets measured '
                     'that warping makes coverage worse.')
ap.add_argument('--save-png', action='store_true', default=True)
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
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')
PRETRAINED, STRUCT_SEED, NOISE_SEED, STEPS, RESCALE_T = \
    'JeffreyXiang/TRELLIS-image-large', 42, 6, 25, 3.0
_ts = np.linspace(1, 0, STEPS + 1); _ts = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
GT = Path(args.frames_dir)
assert args.res == RENDER_RES, (
    f'--res {args.res} != the loss renderer RENDER_RES {RENDER_RES}; the mask '
    f'would not line up with what training renders.')


def encode(dino, i):
    img = Image.open(GT / f'frame_{i:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


def main():
    print('=' * 84)
    print(f'RENDER MASK v1   frames={GT}   ref={args.ref_frame}   res={args.res}')
    print('=' * 84, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(GT / f'frame_{args.ref_frame:04d}.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    n_vox = int(coords.shape[0])
    print(f'[STRUCT] N_vox = {n_vox:,}', flush=True)
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass
    dino = pipe.models['image_cond_model'].to(DEVICE)
    cond = encode(dino, args.ref_frame).unsqueeze(0).to(DEVICE)
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(n_vox, fm.in_channels, device=DEVICE)
    x = sp.SparseTensor(feats=noise, coords=coords)
    with torch.no_grad():
        for t, tp in T_PAIRS:
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            x = x.replace(x.feats - (t - tp) * fm(x, tt, cond).feats)
        mesh = dec(normalize_slat(x))[0]

    v = mesh.vertices.detach(); f = mesh.faces.detach()
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    f = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]

    if args.alignment:
        import math
        A = json.load(open(args.alignment))
        rv = np.asarray(A['rotvec'], float); th = float(np.linalg.norm(rv)) + 1e-12
        k = rv / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = torch.tensor(np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K),
                         dtype=torch.float32, device=DEVICE)
        c = torch.tensor(A['centre'], dtype=torch.float32, device=DEVICE)
        t_ = torch.tensor(A['translation'], dtype=torch.float32, device=DEVICE)
        v = float(A['scale']) * ((v - c) @ R.T) + c + t_
        print(f'[ALIGN] applied scale={A["scale"]:.5f}', flush=True)

    from trellis.representations.mesh import MeshExtractResult
    col = mesh.vertex_attrs[:, :3].detach().clamp(0, 1)
    r = make_renderer(DEVICE)
    res = r.render(MeshExtractResult(vertices=v, faces=f,
                   vertex_attrs=torch.cat([col, col], 1), res=256),
                   EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                   return_types=['mask'])
    m = (res['mask'].squeeze() > 0.5).cpu().numpy()

    frac = m.mean()
    assert 0.02 < frac < 0.60, (
        f'silhouette is {100*frac:.2f}% of frame — the render failed or the '
        f'camera does not see this object.')
    np.save(OUT, m)
    print(f'\n[DONE] {OUT}')
    print(f'  silhouette {int(m.sum()):,} px  ({100*frac:.2f}% of {args.res}^2)')
    json.dump(dict(frames_dir=str(GT), ref_frame=args.ref_frame, res=args.res,
                   n_vox=n_vox, silhouette_px=int(m.sum()),
                   aligned=bool(args.alignment), verts=int(mesh.vertices.shape[0])),
              open(OUT.parent / 'render_mask_meta.json', 'w'), indent=2)
    if args.save_png:
        Image.fromarray((m * 255).astype(np.uint8)).save(OUT.parent / 'render_mask.png')
        print(f'  wrote {OUT.parent / "render_mask.png"}')


if __name__ == '__main__':
    main()
