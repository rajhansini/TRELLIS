"""
render_rung9_comparison.py
--------------------------
GT video | frozen TRELLIS | rung9 (geometry + colour LoRA), all 150 frames, one mp4.

WHY THIS IS NOT render_final_comparison.py
  That script hardcodes the colour-only write window:

      f[:, 53:101] = (f[:, 53:101] + lora(x)).clamp(ABS_MIN, ABS_MAX)

  rung9's B has 80 rows, not 48. Fed through that path the first 48 rows would
  land on colour and the geometry rows would be dropped on the floor — the video
  would show a wrong adapter and look like a milder rung8. Nothing would error.

  This script reads BLOCKS out of the checkpoint ('blocks' / 'write_blocks',
  written by rung9_geomcolor_lora.py) instead of assuming a layout, so it renders
  whatever the checkpoint actually trained, and asserts the row count matches.

The clamp is applied to [53:101] only, exactly as in training — applying a
colour-fitted bound to sdf would render a different model than the one trained.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_rung9_comparison.py
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
_LORA = _HERE.parent
_ROOT = _LORA.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--rung9-run', default='rung9_geom-color_r4_s6_e44d8696')
ap.add_argument('--ckpt',      default='lora_best.pt')
ap.add_argument('--gt-dir',    default=None, type=Path)
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--fps',       type=int, default=15)
ap.add_argument('--abs-min',   type=float, default=-9.0)
ap.add_argument('--abs-max',   type=float, default=8.0)
args = ap.parse_args()

RUN_DIR  = _HERE / 'runs' / args.rung9_run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
GT_DIR   = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (RUN_DIR / 'comparison')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_START, COLOR_END = 53, 101
DEC_OUT_IN_DIM = 96


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
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local, '--no-build-isolation',
             '--no-cache-dir', '--no-deps', '-q'],
            cwd=f'{src}/nvdiffrast', env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')


class OutLayerLoRA(nn.Module):
    def __init__(self, rank, in_dim, out_dim):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def _clamp_colour_only(rw, s, e):
    """Identical to training: clamp the [53:101] portion of a block, nothing else."""
    lo, hi = max(s, COLOR_START), min(e, COLOR_END)
    if lo >= hi:
        return rw
    if lo == s and hi == e:
        return rw.clamp(args.abs_min, args.abs_max)
    out = rw.clone()
    out[:, lo - s:hi - s] = rw[:, lo - s:hi - s].clamp(args.abs_min, args.abs_max)
    return out


def lora_forward(dec_model, lora, blocks, offs, slat):
    def _hook(mod, inp, out):
        delta = lora(inp[0].feats)
        nf = out.feats.clone()
        for (s, e), off in zip(blocks, offs):
            w = e - s
            nf[:, s:e] = _clamp_colour_only(nf[:, s:e] + delta[:, off:off + w], s, e)
        return out.replace(nf)
    with torch.no_grad():
        h = dec_model.out_layer.register_forward_hook(_hook)
        meshes = dec_model(slat)
        h.remove()
    return meshes[0]


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
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
    pil = Image.fromarray(np.full((H + LAB, W * len(arrays), 3), 18, np.uint8))
    dr  = ImageDraw.Draw(pil)
    for i, (a, l) in enumerate(zip(arrays, labels)):
        pil.paste(Image.fromarray(a), (i * W, LAB))
        dr.rectangle([i * W, 0, (i + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
        dr.text((i * W + 10, 11), l, fill=(240, 240, 240))
    return pil


def main():
    assert CKPT.exists(),     f'missing checkpoint: {CKPT}'
    assert SLAT_NPZ.exists(), f'missing SLaT cache: {SLAT_NPZ}'

    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    blocks = [tuple(b) for b in ck['blocks']]
    offs, o = [], 0
    for s, e in blocks:
        offs.append(o); o += e - s
    out_dim = o
    Bw = ck['lora_state']['B']
    rank = Bw.shape[1]
    assert Bw.shape[0] == out_dim, (
        f'checkpoint B has {Bw.shape[0]} rows but blocks {blocks} need {out_dim}. '
        f'Rendering this would silently apply the adapter to the wrong channels.')

    print(f'[CKPT] {CKPT.name}  epoch={ck["epoch"]}  write_blocks={ck["write_blocks"]}')
    print(f'[CKPT] blocks={blocks}  offsets={offs}  rank={rank}  out_dim={out_dim}')
    print(f'[CKPT] ||B||={Bw.float().norm().item():.4f}', flush=True)

    lora = OutLayerLoRA(rank, DEC_OUT_IN_DIM, out_dim).to(DEVICE)
    lora.load_state_dict(ck['lora_state']); lora.eval()

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] {slats.shape}  coords {tuple(coords.shape)} (pinned, shared)', flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec  = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    for n in list(pipe.models.keys()):
        if n != 'slat_decoder_mesh':
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(exist_ok=True)

    a_frz, a_l9, v_frz, v_l9 = [], [], [], []

    for fi in range(1, args.n_frames + 1):
        gt = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png')
                      .convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))

        with torch.no_grad():
            m_f = dec(st)[0]
        nvf = int(m_f.vertices.shape[0])
        rgb_f, ar_f = render(m_f, renderer)

        m_9 = lora_forward(dec, lora, blocks, offs, st)
        nv9 = int(m_9.vertices.shape[0])
        rgb_9, ar_9 = render(m_9, renderer)

        a_frz.append(ar_f); a_l9.append(ar_9); v_frz.append(nvf); v_l9.append(nv9)
        strip([gt, rgb_f, rgb_9],
              ['GT video',
               f'frozen TRELLIS  V={nvf:,}',
               f'rung9 geom+colour  V={nv9:,}']
              ).save(fdir / f'cmp_{fi:04d}.png')

        if fi % 25 == 0 or fi == args.n_frames:
            print(f'  {fi}/{args.n_frames}  frozen V={nvf:,} area={ar_f:,}   '
                  f'rung9 V={nv9:,} area={ar_9:,}', flush=True)
        del st, m_f, m_9
        gc.collect(); torch.cuda.empty_cache()

    def spread(x):
        x = np.array(x, float); return (x.max() - x.min()) / x.mean() * 100

    print(f'\n[SILHOUETTE AREA]')
    print(f'  frozen  spread {spread(a_frz):5.2f}%   verts spread {spread(v_frz):5.2f}%')
    print(f'  rung9   spread {spread(a_l9):5.2f}%   verts spread {spread(v_l9):5.2f}%', flush=True)

    ff = '/usr/bin/ffmpeg'
    pr = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    vid = OUT_DIR / 'GT_vs_frozen_vs_rung9.mp4'
    subprocess.run([ff, '-y', '-framerate', str(args.fps),
                    '-i', str(fdir / 'cmp_%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)', flush=True)

    json.dump(dict(area_frozen=a_frz, area_rung9=a_l9,
                   verts_frozen=v_frz, verts_rung9=v_l9,
                   spread_area_frozen=spread(a_frz), spread_area_rung9=spread(a_l9),
                   spread_verts_frozen=spread(v_frz), spread_verts_rung9=spread(v_l9),
                   ckpt_epoch=ck['epoch'], blocks=blocks),
              open(OUT_DIR / 'comparison.json', 'w'), indent=2)
    print(f'[DONE] {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
