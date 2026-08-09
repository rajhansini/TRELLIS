"""
render_pinned_vs_free.py
------------------------
Visual proof of the pinned-vs-free coords result.

geom_change (pinned) and geom_change_free_coords (free) produced numbers and
plots but saved no frames. This renders both so the flicker can be seen rather
than inferred.

  GT video  |  PINNED coords (frame 75, shared)  |  FREE coords (per frame)

No adapter in either arm — frozen decoder only. The only difference between the
two right-hand panels is where `coords` comes from:

  PINNED  the shared 7,301-voxel structure from the SLaT cache, reused every frame
  FREE    pipeline.run() per frame, so stage B samples the structure from that
          frame's own image.

SEED: both arms run at 42. The cache was built with STRUCT_SEED=42, so the free
arm must use 42 as well — otherwise the two panels differ by seed *and* by coords
and neither can be attributed. Do not change --seed without rebuilding the cache
to match.

What to look for: the FREE panel jumping between discrete shapes, particularly
around frames 11 and 33 where the measured discontinuities were. FlexiCubes calls
this "the isosurface slips over a grid vertex, the mesh jumps discontinuously".

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_pinned_vs_free.py
"""

import sys, os, gc, json, argparse, subprocess
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_LORA = _ROOT / 'experiments' / 'lora_experiments'

ap = argparse.ArgumentParser()
ap.add_argument('--v4-run',   default='rung5_colonly_outlayer_r4_s6_c85c888f')
ap.add_argument('--gt-dir',   default=None, type=Path)
ap.add_argument('--out-dir',  default=None, type=Path)
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--seed',     type=int, default=42,
                help='MUST match STRUCT_SEED used to build slat_cache.npz (42), else the two arms differ by seed as well as by coords')
ap.add_argument('--fps',      type=int, default=15)
args = ap.parse_args()

SLAT_NPZ = _LORA / 'runs' / args.v4_run / 'slat_cache.npz'
GT_DIR   = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (_HERE / 'pinned_vs_free')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p = torch.cuda.get_device_properties(0)
    arch_tag, arch_str = f'sm{p.major}{p.minor}', f'{p.major}.{p.minor}'
    local = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)
    try:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext(); del glctx
        print('[NVDIFF] OK', flush=True); return
    except Exception as e:
        print(f'[NVDIFF] FAILED: {e}', flush=True)
    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast broken for {arch_tag}')
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git','clone','https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast','--depth','1','--quiet'], check=True, env=env)
    _sp.run([pip,'install','.','--target',local,'--no-build-isolation',
             '--no-cache-dir','--no-deps','-q'], cwd=f'{src}/nvdiffrast', env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render(mesh, renderer):
    res = renderer.render(filter_degenerate_faces(mesh),
                          EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
    m = res['mask']
    c = (res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))).detach().clamp(0, 1)
    return ((c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            int((m > 0.5).sum().item()))


LAB = 34
def strip(arrays, labels):
    H, W = arrays[0].shape[:2]
    canvas = np.full((H + LAB, W * len(arrays), 3), 18, np.uint8)
    pil = Image.fromarray(canvas); dr = ImageDraw.Draw(pil)
    for i, (a, l) in enumerate(zip(arrays, labels)):
        pil.paste(Image.fromarray(a), (i * W, LAB))
        dr.rectangle([i*W, 0, (i+1)*W-1, LAB-1], fill=(38, 38, 58))
        dr.text((i*W + 10, 11), l, fill=(240, 240, 240))
    return pil


def main():
    print(f'[SLAT] {SLAT_NPZ}', flush=True)
    raw = np.load(SLAT_NPZ, allow_pickle=True)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] pinned coords: {coords.shape[0]:,} voxels, shared', flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    pipe.to(DEVICE)
    dec = pipe.models['slat_decoder_mesh'].eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(exist_ok=True)

    area_pin, area_free, verts_free = [], [], []

    for fi in range(1, args.n_frames + 1):
        img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
        gt = np.array(img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS))

        # PINNED — decode the cached SLaT on the shared coords
        feats = torch.from_numpy(slats[fi - 1].copy()).float()
        st = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))
        with torch.no_grad():
            m_pin = dec(st)[0]
        rgb_pin, a_pin = render(m_pin, renderer)

        # FREE — full stock run, structure sampled from this frame's image
        out = pipe.run(img, num_samples=1, seed=args.seed, formats=['mesh'])
        m_free = out['mesh'][0]
        nv = int(m_free.vertices.shape[0])
        rgb_free, a_free = render(m_free, renderer)

        area_pin.append(a_pin); area_free.append(a_free); verts_free.append(nv)
        strip([gt, rgb_pin, rgb_free],
              ['GT video',
               'PINNED coords (frame 75, shared)',
               f'FREE coords (per frame)  verts={nv:,}']
              ).save(fdir / f'cmp_{fi:04d}.png')

        if fi % 10 == 0 or fi == args.n_frames:
            print(f'  {fi}/{args.n_frames}  pin_area={a_pin:,}  '
                  f'free_area={a_free:,}  free_verts={nv:,}', flush=True)
        del st, m_pin, out, m_free, feats
        gc.collect(); torch.cuda.empty_cache()

    ap_, af = np.array(area_pin, float), np.array(area_free, float)
    jp = np.abs(np.diff(ap_)).mean(); jf = np.abs(np.diff(af)).mean()
    print(f'\n[SILHOUETTE AREA, frame to frame]')
    print(f'  pinned  mean jump {jp:8.1f} px   spread {(ap_.max()-ap_.min())/ap_.mean()*100:5.2f}%')
    print(f'  free    mean jump {jf:8.1f} px   spread {(af.max()-af.min())/af.mean()*100:5.2f}%')
    print(f'  free jumps {jf/max(jp,1e-9):.1f}x more per frame than pinned', flush=True)

    ff = '/usr/bin/ffmpeg'
    pr = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v','libx264','-crf','18','-pix_fmt','yuv420p'] if 'libx264' in pr.stdout
          else ['-c:v','mpeg4','-q:v','5','-pix_fmt','yuv420p'])
    vid = OUT_DIR / 'PINNED_vs_FREE_coords.mp4'
    subprocess.run([ff,'-y','-framerate',str(args.fps),'-i',str(fdir/'cmp_%04d.png'),
                    '-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)', flush=True)

    json.dump(dict(area_pinned=area_pin, area_free=area_free, verts_free=verts_free,
                   jump_pinned=float(jp), jump_free=float(jf),
                   jump_ratio=float(jf/max(jp,1e-9))),
              open(OUT_DIR / 'pinned_vs_free.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
