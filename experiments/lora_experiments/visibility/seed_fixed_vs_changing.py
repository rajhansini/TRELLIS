"""
seed_fixed_vs_changing.py
-------------------------
Does pinning the random seed remove flicker?

Two arms, 150 frames each, STOCK TRELLIS. The only difference between them is the
value handed to pipeline.run(seed=...):

  FIXED     seed = 6            same seed on every frame
  CHANGING  seed = frame index  a different seed on every frame

Everything else is untouched — same conditioning, same sampler params, same
decoder, preprocess_image left at its shipped default. Any difference between the
arms is attributable to the seed and nothing else.

pipeline.run() calls torch.manual_seed(seed) once, before sampling BOTH the
sparse structure and the SLaT, so a single seed governs both random draws.

OUTPUT
  a 150-frame video: GT | fixed seed | changing seed
  plus jitter/drift and consecutive-silhouette-IoU numbers per arm

READING THE RESULT
  fixed stable + changing jittery -> the seed is what buys stability
  both jittery                    -> the seed is not the flicker source; the
                                     per-frame conditioning is, and only pinning
                                     coords can remove it

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/seed_fixed_vs_changing.py
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

ap = argparse.ArgumentParser()
ap.add_argument('--gt-dir',   default=None, type=Path)
ap.add_argument('--out-dir',  default=None, type=Path)
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--fixed-seed', type=int, default=6)
ap.add_argument('--fps',      type=int, default=15)
args = ap.parse_args()

GT_DIR  = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR = (args.out_dir or (_HERE / 'seed_test')).resolve()
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
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
    rgb = (c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return rgb, (m > 0.5).cpu().numpy()


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


def stats(imgs, masks, nvox, tag):
    n = len(imgs)
    a = np.stack(imgs).astype(np.float32) / 255.0
    d = np.array([np.abs(a[i] - a[i-1]).mean() for i in range(1, n)])
    jitter = d.mean()
    drift = np.abs(a[-1] - a[0]).mean() / (n - 1)
    ratio = jitter / max(drift, 1e-12)
    iou = np.array([(masks[i] & masks[i-1]).sum() / max((masks[i] | masks[i-1]).sum(), 1)
                    for i in range(1, n)])
    nv = np.array(nvox)
    print(f'\n[{tag}]')
    print(f'  jitter (consecutive)        : {jitter:.5f}')
    print(f'  drift  (per frame if smooth): {drift:.5f}')
    print(f'  RATIO jitter/drift          : {ratio:.2f}')
    print(f'  silhouette IoU consecutive  : mean {iou.mean():.4f}  min {iou.min():.4f}')
    print(f'  mesh verts                  : {nv.mean():,.0f} +- {nv.std():,.0f} '
          f'(spread {(nv.max()-nv.min())/nv.mean()*100:.1f}%)', flush=True)
    return dict(arm=tag, jitter=float(jitter), drift=float(drift), ratio=float(ratio),
                iou_mean=float(iou.mean()), iou_min=float(iou.min()),
                verts_mean=float(nv.mean()), verts_std=float(nv.std()),
                verts_spread_pct=float((nv.max()-nv.min())/nv.mean()*100)), d, iou


def main():
    frames = list(range(1, args.n_frames + 1))
    print(f'{len(frames)} frames · fixed seed = {args.fixed_seed} · '
          f'changing seed = frame index', flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    pipe.to(DEVICE)
    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(exist_ok=True)

    R = {'fixed': ([], [], []), 'changing': ([], [], [])}
    for k, fi in enumerate(frames):
        img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
        gt = np.array(img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        panels = [gt]
        for arm, seed in (('fixed', args.fixed_seed), ('changing', fi)):
            out = pipe.run(img, num_samples=1, seed=seed, formats=['mesh'])
            mesh = out['mesh'][0]
            rgb, m = render(mesh, renderer)
            R[arm][0].append(rgb); R[arm][1].append(m)
            R[arm][2].append(int(mesh.vertices.shape[0]))
            panels.append(rgb)
            del out, mesh
            gc.collect(); torch.cuda.empty_cache()
        strip(panels, ['GT video',
                       f'FIXED seed={args.fixed_seed}',
                       f'CHANGING seed={fi}']).save(fdir / f'cmp_{fi:04d}.png')
        if fi % 10 == 0 or fi == frames[-1]:
            print(f'  {fi}/{len(frames)}', flush=True)

    sF, dF, uF = stats(*R['fixed'],    'FIXED seed')
    sC, dC, uC = stats(*R['changing'], 'CHANGING seed')

    print(f'\n{"="*70}\n[VERDICT]\n{"="*70}')
    print(f'  flicker ratio   fixed {sF["ratio"]:.2f}   changing {sC["ratio"]:.2f}')
    print(f'  silhouette IoU  fixed {sF["iou_mean"]:.4f}  changing {sC["iou_mean"]:.4f}')
    print(f'  vertex spread   fixed {sF["verts_spread_pct"]:.1f}%  '
          f'changing {sC["verts_spread_pct"]:.1f}%')
    if sF['ratio'] < 2.0 and sC['ratio'] > sF['ratio'] * 1.5:
        v = 'PINNING THE SEED REMOVES THE FLICKER.'
    elif sF['ratio'] > 3.0:
        v = ('flicker REMAINS with the seed pinned — the source is the per-frame '
             'conditioning, not the random draw. Only pinning coords can remove it.')
    else:
        v = 'seed pinning helps but does not fully remove the flicker.'
    print(f'  -> {v}', flush=True)

    ff = '/usr/bin/ffmpeg'
    pr = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v','libx264','-crf','18','-pix_fmt','yuv420p'] if 'libx264' in pr.stdout
          else ['-c:v','mpeg4','-q:v','5','-pix_fmt','yuv420p'])
    vid = OUT_DIR / 'SEED_fixed_vs_changing.mp4'
    subprocess.run([ff,'-y','-framerate',str(args.fps),'-i',str(fdir/'cmp_%04d.png'),
                    '-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)', flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Fixed vs changing seed — stock TRELLIS, 150 frames')
    x = frames[1:]
    ax[0].plot(x, dF, '-', color='#61afef', label=f'fixed (ratio {sF["ratio"]:.1f})')
    ax[0].plot(x, dC, '-', color='#e06c75', label=f'changing (ratio {sC["ratio"]:.1f})')
    ax[0].set_xlabel('frame'); ax[0].set_ylabel('|I_t - I_{t-1}|')
    ax[0].set_title('Render jitter'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, uF, '-', color='#61afef'); ax[1].plot(x, uC, '-', color='#e06c75')
    ax[1].set_xlabel('frame'); ax[1].set_ylabel('IoU(t, t-1)')
    ax[1].set_title('Silhouette stability'); ax[1].grid(alpha=.3)
    ax[2].plot(frames, R['fixed'][2], '-', color='#61afef')
    ax[2].plot(frames, R['changing'][2], '-', color='#e06c75')
    ax[2].set_xlabel('frame'); ax[2].set_ylabel('mesh vertices')
    ax[2].set_title('Sparse structure stability'); ax[2].grid(alpha=.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'seed_fixed_vs_changing.png', dpi=140, bbox_inches='tight')
    json.dump({'fixed': sF, 'changing': sC, 'verdict': v,
               'verts_fixed': R['fixed'][2], 'verts_changing': R['changing'][2]},
              open(OUT_DIR / 'seed_fixed_vs_changing.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
