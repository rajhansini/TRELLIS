"""
geom_change_free_coords.py
--------------------------
Re-measures the geometry tracking ratio with FREE per-frame coords.

WHY
  measure_geometry_change.py gave a tracking ratio of 0.063 — TRELLIS's mesh moves
  0.3% while the GT video's teapot moves 5.4%. That number was measured with the
  sparse structure PINNED to frame 75 and reused for all 150 frames.

  Free the coords and stage B samples the occupancy from each frame's own image,
  so the coarse shape may follow the video by itself with no adapter involved.
  That number has never been measured. It decides whether a geometry LoRA is
  needed at all.

WHAT IS DIFFERENT FROM measure_geometry_change.py
  pinned version : loads slat_cache.npz (shared frame-75 coords), decodes only
  this version   : runs stock TRELLIS end to end per frame, so coords are sampled
                   fresh from each frame's image. Cannot reuse the cache — the
                   cache was built on the shared coords.

  Seed is held at 6 on every frame so the random draw is constant and the ONLY
  thing varying between frames is the image conditioning. Otherwise a changing
  shape could be the dice rather than the video.

METRIC (identical to the pinned version, so the two are directly comparable)
  silhouette area and IoU vs frame 1, at 0/90/180/270 degrees
  chamfer distance to the frame-1 mesh, subsampled
  the GT video's own silhouette change, tight mask min(RGB) < 0.95

  tracking ratio = TRELLIS silhouette spread / GT silhouette spread
     ~1.0  stage B already tracks the shape -> no geometry LoRA needed
     ~0.0  the shape signal never arrives   -> geometry LoRA justified
     pinned baseline for comparison: 0.063

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/geom_change_free_coords.py
"""

import sys, os, gc, json, math, argparse
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
ap.add_argument('--gt-dir',    default=None, type=Path)
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--stride',    type=int, default=1)
ap.add_argument('--seed',      type=int, default=6,
                help='held constant so only the image conditioning varies')
ap.add_argument('--n-sample',  type=int, default=20000,
                help='vertices subsampled per mesh for chamfer')
ap.add_argument('--gt-thresh', type=float, default=0.95)
args = ap.parse_args()

GT_DIR  = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR = (args.out_dir or (_HERE / 'diagnostics' / 'geomchange_free_coords')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORBIT_ANGLES = [0.0, 90.0, 180.0, 270.0]
PINNED_RATIO = 0.063          # from measure_geometry_change.py, for reference


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


def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    """theta=0 reproduces the confirmed front-view EXTRINSICS."""
    theta, el = math.radians(theta_deg), math.radians(elevation_deg)
    C = torch.tensor([radius * math.sin(theta) * math.cos(el),
                      -radius * math.cos(theta) * math.cos(el),
                      radius * math.sin(el)], dtype=torch.float32)
    up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cz = -C / C.norm()
    cx = torch.linalg.cross(up, cz); cx = cx / cx.norm()
    cy = torch.linalg.cross(cz, cx); cy = cy / cy.norm()
    R = torch.stack([cx, cy, cz], 0)
    E = torch.eye(4); E[:3, :3] = R; E[:3, 3] = R @ (-C)
    return E.to(DEVICE)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render_mask(mesh, renderer, ext):
    res = renderer.render(mesh, ext, INTRINSICS.to(DEVICE), return_types=['mask'])
    return res['mask'] > 0.5


def chamfer(a, b, chunk=2048):
    def one_way(x, y):
        tot, n = 0.0, 0
        for i in range(0, x.shape[0], chunk):
            d = torch.cdist(x[i:i+chunk], y)
            tot += d.min(dim=1).values.sum().item(); n += d.shape[0]
        return tot / max(n, 1)
    return 0.5 * (one_way(a, b) + one_way(b, a))


def subsample(v, n, gen):
    if v.shape[0] <= n:
        return v
    idx = torch.randperm(v.shape[0], generator=gen)[:n]
    return v[idx.to(v.device)]


def gt_mask(fi):
    im = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
    im = im.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    a = torch.from_numpy(np.array(im)).float().div(255.).permute(2, 0, 1)
    return (a.min(dim=0).values < args.gt_thresh)


def iou(a, b):
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


def main():
    frames = list(range(1, args.n_frames + 1, args.stride))
    print('=' * 78)
    print('Geometry tracking with FREE per-frame coords')
    print(f'  frames : {len(frames)}   seed held at {args.seed} (only the image varies)')
    print(f'  pinned-coords baseline for comparison: tracking ratio {PINNED_RATIO}')
    print('=' * 78, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    pipe.to(DEVICE)
    renderer = make_renderer(DEVICE)
    exts = {th: orbit_extrinsics(th) for th in ORBIT_ANGLES}
    gen = torch.Generator().manual_seed(0)

    ref_pts, ref_masks, ref_gt, noise_floor = None, {}, None, None
    rows = []
    print(f'\n{"frame":>6} {"verts":>9} {"coord_vox":>10} {"chamfer_f1":>12} '
          f'{"sil_0deg":>10} {"IoU_f1":>8} {"GT_area":>9} {"GT_IoU":>8}')

    for fi in frames:
        img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
        out = pipe.run(img, num_samples=1, seed=args.seed, formats=['mesh'])
        mesh = filter_degenerate_faces(out['mesh'][0])

        v = mesh.vertices.detach().float()
        pts = subsample(v, args.n_sample, gen)
        masks = {th: render_mask(mesh, renderer, exts[th]) for th in ORBIT_ANGLES}
        gm = gt_mask(fi).to(DEVICE)

        if ref_pts is None:
            ref_pts = pts.clone()
            ref_masks = {k: m.clone() for k, m in masks.items()}
            ref_gt = gm.clone()
            noise_floor = chamfer(ref_pts, ref_pts.clone())
            print(f'[NOISE] chamfer(f1,f1) = {noise_floor:.3e} — anything at or below '
                  f'this is round-off, not motion', flush=True)

        cd = chamfer(pts, ref_pts) if fi != frames[0] else 0.0
        row = dict(frame=fi, n_verts=int(v.shape[0]),
                   n_faces=int(mesh.faces.shape[0]),
                   chamfer_vs_f1=float(cd),
                   bbox=[float(x) for x in (v.max(0).values - v.min(0).values).tolist()],
                   sil_area={str(t): int(masks[t].sum()) for t in ORBIT_ANGLES},
                   sil_iou_f1={str(t): float(iou(masks[t], ref_masks[t])) for t in ORBIT_ANGLES},
                   gt_area=int(gm.sum()), gt_iou_f1=float(iou(gm, ref_gt)))
        rows.append(row)
        print(f'{fi:>6} {row["n_verts"]:>9,} {"—":>10} {cd:>12.6f} '
              f'{row["sil_area"]["0.0"]:>10,} {row["sil_iou_f1"]["0.0"]:>8.5f} '
              f'{row["gt_area"]:>9,} {row["gt_iou_f1"]:>8.5f}', flush=True)

        del out, mesh, v, pts, masks, gm
        gc.collect(); torch.cuda.empty_cache()

    a_tr = np.array([r['sil_area']['0.0'] for r in rows], float)
    a_gt = np.array([r['gt_area'] for r in rows], float)
    nv   = np.array([r['n_verts'] for r in rows], float)
    d_tr = (a_tr[-1] - a_tr[0]) / a_tr[0] * 100
    d_gt = (a_gt[-1] - a_gt[0]) / a_gt[0] * 100
    rng_tr = (a_tr.max() - a_tr.min()) / a_tr.mean() * 100
    rng_gt = (a_gt.max() - a_gt.min()) / a_gt.mean() * 100
    iou_tr = min(r['sil_iou_f1']['0.0'] for r in rows)
    iou_gt = min(r['gt_iou_f1'] for r in rows)
    cd_max = max(r['chamfer_vs_f1'] for r in rows)
    bbox0  = max(rows[0]['bbox'])
    ratio  = rng_tr / rng_gt if rng_gt > 1e-9 else float('nan')
    # flicker: how much does the mesh jump frame to frame vs the trend
    jit = np.abs(np.diff(a_tr)).mean()
    drf = abs(a_tr[-1] - a_tr[0]) / max(len(a_tr) - 1, 1)
    flick = jit / max(drf, 1e-12)

    print('\n' + '=' * 78 + '\n[VERDICT]\n' + '=' * 78)
    print('  TRELLIS, free per-frame coords:')
    print(f'    silhouette area f1 -> f{frames[-1]}  : {d_tr:+.2f}%')
    print(f'    silhouette area spread          : {rng_tr:.2f}% of mean')
    print(f'    lowest silhouette IoU vs f1     : {iou_tr:.5f}')
    print(f'    max chamfer to f1               : {cd_max:.6f} '
          f'({cd_max/bbox0*100:.3f}% of bbox, {cd_max/max(noise_floor,1e-12):.0f}x noise)')
    print(f'    mesh verts                      : {nv.mean():,.0f} +- {nv.std():,.0f} '
          f'(spread {(nv.max()-nv.min())/nv.mean()*100:.1f}%)')
    print('  GT video:')
    print(f'    silhouette area f1 -> f{frames[-1]}  : {d_gt:+.2f}%')
    print(f'    silhouette area spread          : {rng_gt:.2f}% of mean')
    print(f'    lowest silhouette IoU vs f1     : {iou_gt:.5f}')
    print(f'\n  TRACKING RATIO (free coords)      = {ratio:.3f}')
    print(f'  TRACKING RATIO (pinned, baseline) = {PINNED_RATIO:.3f}')
    print(f'  improvement                       = {ratio/PINNED_RATIO:.1f}x')
    print(f'\n  FLICKER (frame-to-frame jump / trend) = {flick:.2f}')

    if ratio > 0.6:
        v = ('STAGE B ALREADY TRACKS THE SHAPE with free coords. A geometry LoRA is '
             'not needed for coarse shape — the remaining problem is temporal coherence.')
    elif ratio < 0.2:
        v = ('FREEING THE COORDS DOES NOT HELP — the shape signal does not arrive even '
             'when the structure is sampled per frame. A geometry LoRA is justified.')
    else:
        v = ('PARTIAL — free coords recovers some shape tracking but not all. A geometry '
             'LoRA would add to it rather than replace it.')
    print(f'  -> {v}')
    if flick > 3.0:
        print(f'  !! flicker ratio {flick:.1f} — the mesh jumps far more than the trend '
              f'requires. Free coords costs temporal coherence.')
    print('=' * 78, flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    fig.suptitle('Geometry tracking — free per-frame coords vs the GT video')
    f = [r['frame'] for r in rows]
    ax[0].plot(f, a_tr / a_tr[0], '-', color='#61afef', label=f'TRELLIS free coords')
    ax[0].plot(f, a_gt / a_gt[0], '-', color='#e06c75', label='GT video')
    ax[0].set_xlabel('frame'); ax[0].set_ylabel('silhouette area / frame 1')
    ax[0].set_title(f'Silhouette area (ratio {ratio:.2f}, pinned was {PINNED_RATIO:.2f})')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for th, c in zip(ORBIT_ANGLES, ('#98c379', '#61afef', '#d19a66', '#c678dd')):
        ax[1].plot(f, [r['sil_iou_f1'][str(th)] for r in rows], '-', color=c,
                   label=f'{th:.0f} deg')
    ax[1].plot(f, [r['gt_iou_f1'] for r in rows], 'k--', lw=2, label='GT (front)')
    ax[1].set_xlabel('frame'); ax[1].set_ylabel('silhouette IoU vs frame 1')
    ax[1].set_title('Shape drift, multi-view'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[2].plot(f, nv, '-', color='#c678dd')
    ax[2].set_xlabel('frame'); ax[2].set_ylabel('mesh vertices')
    ax[2].set_title(f'Structure stability (flicker {flick:.1f})'); ax[2].grid(alpha=.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'geom_change_free_coords.png', dpi=140, bbox_inches='tight')
    print(f'[SAVE] {OUT_DIR / "geom_change_free_coords.png"}')

    json.dump(dict(mode='free_per_frame_coords', seed=args.seed, frames=f, rows=rows,
                   trellis_area_change_pct=d_tr, gt_area_change_pct=d_gt,
                   trellis_area_spread_pct=rng_tr, gt_area_spread_pct=rng_gt,
                   min_iou_trellis=iou_tr, min_iou_gt=iou_gt,
                   max_chamfer=cd_max, chamfer_noise_floor=noise_floor,
                   tracking_ratio=ratio, tracking_ratio_pinned=PINNED_RATIO,
                   flicker_ratio=flick, verdict=v),
              open(OUT_DIR / 'geom_change_free_coords.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "geom_change_free_coords.json"}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
