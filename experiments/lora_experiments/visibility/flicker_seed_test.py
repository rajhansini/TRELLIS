"""
flicker_seed_test.py
--------------------
Does fixing the random seed alone remove flicker, or do you also need to pin the
sparse structure?

STOCK TRELLIS ONLY. No adapter, no SLaT cache, no pinned coords — we call
pipeline.run() exactly as shipped. run() already does torch.manual_seed(seed)
before sampling both the sparse structure and the SLaT, so the two arms differ
by one argument and nothing else:

  arm A  seed = frame index   fresh dice every frame  (stock behaviour)
  arm B  seed = 42            same dice every frame

Everything else — conditioning, sampler params, decoder — is identical. Any
difference between the arms is attributable to the seed alone.

METRIC
  jitter = mean |I_t - I_{t-1}|                 frame-to-frame change
  drift  = |I_last - I_first| / (n_steps - 1)   per-frame change if it were smooth
  ratio  = jitter / drift

  ratio ~ 1   smooth evolution, no flicker
  ratio  5    frames jump 5x more than the trend requires
  ratio >> 10 flicker dominates

Computed on the render, and separately on the silhouette (consecutive IoU) which
isolates geometry popping from colour flicker. Voxel counts are logged too — if
they swing between frames, the sparse structure is unstable regardless of seed.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/flicker_seed_test.py
"""

import sys, os, math, gc, json, argparse
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
ap.add_argument('--gt-dir', default=None, type=Path)
ap.add_argument('--out-dir', default=None, type=Path)
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--stride',   type=int, default=1, help='every frame — flicker is a consecutive-frame quantity, subsampling invalidates it')
ap.add_argument('--seed',     type=int, default=42, help='the pinned seed for arm B')
args = ap.parse_args()

GT_DIR  = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR = (args.out_dir or (_HERE / 'diagnostics' / 'flicker_seed_test')).resolve()
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
from PIL import Image
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
    return c.cpu().numpy(), (m > 0.5).cpu().numpy()


def run_arm(pipe, renderer, frames, seed_fn, tag):
    """seed_fn(frame_index) -> the seed handed to pipeline.run()"""
    print(f'\n{"="*70}\n[ARM {tag}]  seed policy: {seed_fn.__doc__}\n{"="*70}', flush=True)
    imgs, masks, nvox = [], [], []
    for k, fi in enumerate(frames):
        img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
        s = seed_fn(fi)
        out = pipe.run(img, num_samples=1, seed=s,
                       formats=['mesh'], preprocess_image=False)
        mesh = out['mesh'][0]
        c, m = render(mesh, renderer)
        imgs.append(c); masks.append(m); nvox.append(int(mesh.vertices.shape[0]))
        if (k + 1) % 5 == 0 or k == len(frames) - 1:
            print(f'  {k+1}/{len(frames)}  frame {fi:03d}  seed={s}  '
                  f'verts={nvox[-1]:,}', flush=True)
        del out, mesh
        gc.collect(); torch.cuda.empty_cache()
    return np.stack(imgs), np.stack(masks), np.array(nvox)


def analyse(imgs, masks, nvox, tag):
    n = len(imgs)
    d_consec = np.array([np.abs(imgs[i] - imgs[i-1]).mean() for i in range(1, n)])
    jitter = d_consec.mean()
    drift = np.abs(imgs[-1] - imgs[0]).mean() / (n - 1)
    ratio = jitter / max(drift, 1e-12)
    iou = np.array([ (masks[i] & masks[i-1]).sum() / max((masks[i] | masks[i-1]).sum(), 1)
                     for i in range(1, n) ])
    r = dict(arm=tag, n=n,
             jitter=float(jitter), drift=float(drift), ratio=float(ratio),
             iou_mean=float(iou.mean()), iou_min=float(iou.min()),
             verts_mean=float(nvox.mean()), verts_std=float(nvox.std()),
             verts_spread_pct=float((nvox.max()-nvox.min())/nvox.mean()*100))
    print(f'\n[{tag}]')
    print(f'  render jitter (consecutive)  : {jitter:.5f}')
    print(f'  render drift  (per frame)    : {drift:.5f}')
    print(f'  RATIO jitter/drift           : {ratio:.2f}')
    print(f'  silhouette IoU consecutive   : mean {iou.mean():.4f}  min {iou.min():.4f}')
    print(f'  mesh verts                   : {nvox.mean():,.0f} +- {nvox.std():,.0f} '
          f'(spread {r["verts_spread_pct"]:.1f}%)', flush=True)
    return r, d_consec, iou


def main():
    frames = list(range(1, args.n_frames + 1, args.stride))
    print(f'{len(frames)} frames, stride {args.stride}: {frames[:5]} ... {frames[-1]}')
    print(f'GT: {GT_DIR}', flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    pipe.to(DEVICE)
    renderer = make_renderer(DEVICE)

    def varying(fi):
        """seed = frame index -> fresh dice every frame (stock behaviour)"""
        return int(fi)

    def pinned(fi):
        """seed = fixed constant -> same dice every frame"""
        return int(args.seed)

    iA, mA, vA = run_arm(pipe, renderer, frames, varying, 'A  varying seed')
    rA, dA, uA = analyse(iA, mA, vA, 'A  varying seed')
    del iA; gc.collect(); torch.cuda.empty_cache()

    iB, mB, vB = run_arm(pipe, renderer, frames, pinned, 'B  pinned seed')
    rB, dB, uB = analyse(iB, mB, vB, 'B  pinned seed')

    print(f'\n{"="*70}\n[VERDICT]\n{"="*70}')
    print(f'  flicker ratio   varying {rA["ratio"]:.2f}   pinned {rB["ratio"]:.2f}'
          f'   -> {rA["ratio"]/max(rB["ratio"],1e-9):.2f}x reduction')
    print(f'  silhouette IoU  varying {rA["iou_mean"]:.4f}  pinned {rB["iou_mean"]:.4f}')
    print(f'  vertex spread   varying {rA["verts_spread_pct"]:.1f}%  '
          f'pinned {rB["verts_spread_pct"]:.1f}%')
    if rB['ratio'] < 2.0:
        v = 'PINNING THE SEED IS SUFFICIENT — no flicker left to fix.'
    elif rB['ratio'] < rA['ratio'] * 0.5:
        v = ('seed pinning helps substantially but flicker REMAINS — the residual '
             'comes from the sparse structure varying with the conditioning, which '
             'only pinning coords can remove.')
    else:
        v = ('seed pinning does NOT solve flicker — the instability is driven by the '
             'per-frame conditioning, not the random draw.')
    print(f'  -> {v}', flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Does pinning the seed remove flicker?  (stock TRELLIS, no adapter)')
    x = frames[1:]
    ax[0].plot(x, dA, 'o-', color='#e06c75', label=f'varying (ratio {rA["ratio"]:.1f})')
    ax[0].plot(x, dB, 'o-', color='#61afef', label=f'pinned (ratio {rB["ratio"]:.1f})')
    ax[0].set_xlabel('frame'); ax[0].set_ylabel('|I_t - I_{t-1}|')
    ax[0].set_title('Render jitter'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, uA, 'o-', color='#e06c75'); ax[1].plot(x, uB, 'o-', color='#61afef')
    ax[1].set_xlabel('frame'); ax[1].set_ylabel('IoU(t, t-1)')
    ax[1].set_title('Silhouette stability'); ax[1].grid(alpha=.3)
    ax[2].plot(frames, vA, 'o-', color='#e06c75'); ax[2].plot(frames, vB, 'o-', color='#61afef')
    ax[2].set_xlabel('frame'); ax[2].set_ylabel('mesh vertices')
    ax[2].set_title('Sparse structure stability'); ax[2].grid(alpha=.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'flicker_seed_test.png', dpi=140, bbox_inches='tight')
    print(f'\n[SAVE] {OUT_DIR / "flicker_seed_test.png"}')
    json.dump({'frames': frames, 'varying': rA, 'pinned': rB, 'verdict': v},
              open(OUT_DIR / 'flicker_seed_test.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "flicker_seed_test.json"}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
