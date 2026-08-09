"""
render_final_comparison.py
--------------------------
The money figure: GT | frozen TRELLIS | v4 | rung8, all 150 frames, as one mp4.

  GT              the Kling video frame
  frozen TRELLIS  the backbone with no adapter at all (baseline)
  v4              union loss  -> the white cast
  rung8           intersection loss -> the fix

One decoder forward per frame; the two adapters differ only in their weights, so
both LoRA meshes are produced from the same captured out_layer input.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_final_comparison.py
"""

import sys, os, math, gc, argparse, subprocess
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
ap.add_argument('--v4-run',    default='rung5_colonly_outlayer_r4_s6_c85c888f')
ap.add_argument('--rung8-run', default='rung8_intersect_r4_s6_1b0d1d64')
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--fps',       type=int, default=15)
ap.add_argument('--out-dir',   default=None, type=Path)
args = ap.parse_args()

V4_CKPT    = _LORA / 'runs' / args.v4_run    / 'lora_ckpts' / 'lora_best.pt'
R8_CKPT    = _HERE / 'runs' / args.rung8_run / 'lora_ckpts' / 'lora_best.pt'
SLAT_NPZ   = _LORA / 'runs' / args.v4_run    / 'slat_cache.npz'
GT_DIR     = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                  '/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR    = (args.out_dir or (_HERE / 'final_comparison')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_OUT_IN_DIM, COLOR_START, COLOR_END = 96, 53, 101
COLOR_DIM = COLOR_END - COLOR_START
ABS_MIN, ABS_MAX = -9.0, 8.0


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
    def __init__(self, rank=4, in_dim=DEC_OUT_IN_DIM, color_dim=COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def load_lora(path, name):
    m = OutLayerLoRA().to(DEVICE)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck['lora_state']); m.eval()
    print(f'[LORA] {name}: epoch={ck.get("epoch")} '
          f'best_psnr={ck.get("best_psnr", float("nan")):.3f}', flush=True)
    return m


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:,1]] - v[f[:,0]]; e2 = v[f[:,2]] - v[f[:,0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render(mesh, renderer):
    res = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color','mask'])
    m = res['mask'].unsqueeze(0)
    c = (res['color'] * m + (1.0 - m)).detach().clamp(0,1)
    return (c.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)


def decode_all(dec_model, loras, feats, coords):
    """One forward; frozen mesh plus one mesh per adapter."""
    st, cap = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE)), {}
    def _hook(mod, inp, out):
        cap['x'] = inp[0].feats; cap['out'] = out
    with torch.no_grad():
        h = dec_model.out_layer.register_forward_hook(_hook)
        frozen = filter_degenerate_faces(dec_model(st)[0])
        h.remove()
        out_h, x = cap['out'], cap['x']
        meshes = []
        for lora in loras:
            f = out_h.feats.clone()
            f[:, COLOR_START:COLOR_END] = (
                f[:, COLOR_START:COLOR_END] + lora(x)).clamp(ABS_MIN, ABS_MAX)
            meshes.append(filter_degenerate_faces(
                dec_model.to_representation(out_h.replace(f))[0]))
    return frozen, meshes


LAB = 34
def strip(arrays, labels):
    H, W = arrays[0].shape[:2]
    canvas = np.full((H + LAB, W * len(arrays), 3), 18, np.uint8)
    pil = Image.fromarray(canvas); dr = ImageDraw.Draw(pil)
    for i, (a, l) in enumerate(zip(arrays, labels)):
        pil.paste(Image.fromarray(a), (i*W, LAB))
        dr.rectangle([i*W, 0, (i+1)*W-1, LAB-1], fill=(38,38,58))
        dr.text((i*W + 10, 11), l, fill=(240,240,240))
    return pil


def main():
    print(f'[CKPT] v4    : {V4_CKPT}\n[CKPT] rung8 : {R8_CKPT}', flush=True)
    assert V4_CKPT.exists() and R8_CKPT.exists() and SLAT_NPZ.exists()

    raw = np.load(SLAT_NPZ, allow_pickle=True)
    slats, coords = raw['slats'], torch.from_numpy(raw['coords'].copy()).int()

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec.parameters(): p.requires_grad_(False)
    for n in list(pipe.models):
        if n != 'slat_decoder_mesh':
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    v4  = load_lora(V4_CKPT, 'v4 (union loss)')
    r8  = load_lora(R8_CKPT, 'rung8 (intersection loss)')
    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(exist_ok=True)

    for fi in range(1, args.n_frames + 1):
        feats = torch.from_numpy(slats[fi-1].copy()).float()
        frozen, (m_v4, m_r8) = decode_all(dec, [v4, r8], feats, coords)
        gt = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png')
                      .convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        a = strip([gt, render(frozen, renderer), render(m_v4, renderer), render(m_r8, renderer)],
                  ['GT video',
                   'frozen TRELLIS  (baseline, no adapter)',
                   'v4  union loss  (white cast)',
                   'rung8  intersection loss  (ours)'])
        a.save(fdir / f'cmp_{fi:04d}.png')
        if fi % 25 == 0 or fi == args.n_frames:
            print(f'  {fi}/{args.n_frames}', flush=True)
        del frozen, m_v4, m_r8, feats
        gc.collect(); torch.cuda.empty_cache()

    ff = '/usr/bin/ffmpeg'
    probe = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    flags = (['-c:v','libx264','-crf','18','-pix_fmt','yuv420p'] if 'libx264' in probe.stdout
             else ['-c:v','mpeg4','-q:v','5','-pix_fmt','yuv420p'])
    out = OUT_DIR / 'FINAL_gt_frozen_v4_rung8.mp4'
    subprocess.run([ff,'-y','-framerate',str(args.fps),'-i',str(fdir/'cmp_%04d.png'),
                    '-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2', *flags, str(out)], check=True)
    print(f'\n[VIDEO] {out}  ({out.stat().st_size/1e6:.1f} MB)', flush=True)
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
