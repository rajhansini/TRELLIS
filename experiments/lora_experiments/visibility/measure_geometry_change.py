"""
measure_geometry_change.py
--------------------------
Answers: does the FROZEN TRELLIS decoder already give time-varying geometry,
or is the shape effectively static across the sequence?

WHY THIS MATTERS
  Geometry channels [0:53] are decoded from the PER-FRAME SLaT, so in principle
  the frozen decoder already outputs a mesh that changes shape every frame — no
  adapter involved. If that is true and the change matches the video, a geometry
  LoRA is solving a solved problem. If the frozen mesh is nearly static while the
  video's object visibly changes shape, the SLaT is not carrying shape and a
  geometry LoRA is justified.

  Note the hard limit either way: the sparse structure (which voxels exist) is
  sampled ONCE (seed 42, from frame 75) and the same coords are reused for all
  150 frames. Shape can move within that shell; the object cannot grow past it
  or break into disconnected pieces.

WHAT IT MEASURES, per frame, on the FROZEN decoder (no LoRA anywhere)
  A. mesh statistics      vertex count, face count, bbox extents
  B. 3D motion            chamfer distance to the frame-1 mesh (subsampled).
                          This is the honest "did the surface actually move in 3D"
                          number — vertex counts differ between frames because
                          FlexiCubes re-extracts topology, so a vertex-to-vertex
                          diff is meaningless.
  C. silhouette           area and IoU vs frame 1, from the front camera AND from
                          90/180/270 degrees. Multi-view catches shape change that
                          is invisible head-on.
  D. the GT video         silhouette area and IoU vs frame 1, using the tight mask
                          min(RGB) < 0.95 (the Kling background is off-white, so
                          the naive <0.99 test returns 99.7% of the image).

THE DECISION NUMBER
  ratio = (TRELLIS silhouette area change) / (GT silhouette area change)
    ~1.0  -> TRELLIS already tracks the video's shape change. No geometry LoRA.
    ~0.0  -> the SLaT carries no shape change. A geometry LoRA is justified.
  If the GT itself barely changes, this dataset cannot test geometry dynamics at
  all and you need a video whose subject actually deforms.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
    experiments/lora_experiments/visibility/measure_geometry_change.py \
    --run-id c85c888f --stride 10 \
    2>&1 | tee experiments/lora_experiments/visibility/logs/geom_change.log
"""

import sys, os, json, gc, math, argparse
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
ap.add_argument('--run-id',    default='c85c888f',
                help='run whose slat_cache.npz is read (frozen decode only)')
ap.add_argument('--run-dir',   default=None, type=Path)
ap.add_argument('--gt-dir',    default=None, type=Path,
                help='GT frame dir (default: the teapot lava frames)')
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--stride',    type=int, default=10, help='probe every Nth frame')
ap.add_argument('--n-sample',  type=int, default=20000,
                help='vertices subsampled per mesh for the chamfer distance')
ap.add_argument('--gt-thresh', type=float, default=0.95)
args = ap.parse_args()

RUN_DIR  = args.run_dir or (_LORA / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}')
RUN_DIR  = Path(RUN_DIR).resolve()
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
GT_DIR   = args.gt_dir or Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (_HERE / 'diagnostics' / f'{args.run_id}_geomchange')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORBIT_ANGLES = [0.0, 90.0, 180.0, 270.0]


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
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag}')
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local, '--no-build-isolation',
             '--no-cache-dir', '--no-deps', '-q'], cwd=f'{src}/nvdiffrast',
            env=env, check=True)
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
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')


def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    """theta=0 reproduces the confirmed front-view EXTRINSICS."""
    theta, el = math.radians(theta_deg), math.radians(elevation_deg)
    C = torch.tensor([radius * math.sin(theta) * math.cos(el),
                      -radius * math.cos(theta) * math.cos(el),
                      radius * math.sin(el)], dtype=torch.float32)
    world_up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cam_z = -C / C.norm()
    cam_x = torch.linalg.cross(world_up, cam_z); cam_x = cam_x / cam_x.norm()
    cam_y = torch.linalg.cross(cam_z, cam_x);    cam_y = cam_y / cam_y.norm()
    R = torch.stack([cam_x, cam_y, cam_z], dim=0)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3,  3] = R @ (-C)
    return E.to(DEVICE)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def render_mask(mesh, renderer, ext):
    res = renderer.render(mesh, ext, INTRINSICS.to(DEVICE), return_types=['mask'])
    return res['mask'] > 0.5


def chamfer(a, b, chunk=2048):
    """Symmetric mean nearest-neighbour distance between two point sets."""
    def one_way(x, y):
        tot, n = 0.0, 0
        for i in range(0, x.shape[0], chunk):
            d = torch.cdist(x[i:i + chunk], y)          # (chunk, M)
            tot += d.min(dim=1).values.sum().item()
            n += d.shape[0]
        return tot / max(n, 1)
    return 0.5 * (one_way(a, b) + one_way(b, a))


def subsample(v, n, gen):
    if v.shape[0] <= n:
        return v
    idx = torch.randperm(v.shape[0], generator=gen, device='cpu')[:n]
    return v[idx.to(v.device)]


def gt_mask(frame_idx):
    img = Image.open(GT_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    a = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1)
    return (a.min(dim=0).values < args.gt_thresh)


def iou(a, b):
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


def main():
    print('=' * 78)
    print('Does the FROZEN decoder already give time-varying geometry?')
    print(f'  slat cache : {SLAT_NPZ}')
    print(f'  GT frames  : {GT_DIR}')
    print(f'  frames     : 1..{args.n_frames} step {args.stride}')
    print(f'  views      : {ORBIT_ANGLES} deg')
    print('=' * 78, flush=True)

    assert SLAT_NPZ.exists(), f'missing SLaT cache: {SLAT_NPZ}'
    assert GT_DIR.exists(),   f'missing GT frames: {GT_DIR}'

    raw       = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr = raw['slats']
    coords_t  = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] {slats_arr.shape}   coords shared across frames: '
          f'{raw["coords"].shape}', flush=True)

    pipe      = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec_model = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec_model.parameters():
        p.requires_grad_(False)
    for name in list(pipe.models.keys()):
        if name != 'slat_decoder_mesh':
            try: pipe.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    renderer = make_renderer(DEVICE)
    exts = {th: orbit_extrinsics(th, 0.0, 2.0) for th in ORBIT_ANGLES}
    gen  = torch.Generator().manual_seed(0)

    frames = list(range(1, args.n_frames + 1, args.stride))
    if frames[-1] != args.n_frames:
        frames.append(args.n_frames)

    ref_pts, ref_masks, ref_gt = None, {}, None
    rows = []

    print(f'\n{"frame":>6} {"verts":>9} {"faces":>9} {"chamfer_vs_f1":>14} '
          f'{"sil_area_0deg":>14} {"IoU_vs_f1_0deg":>15} {"GT_area":>9} {"GT_IoU":>8}')

    for fi in frames:
        feats = torch.from_numpy(slats_arr[fi - 1].copy()).float()
        st = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords_t.to(DEVICE))
        with torch.no_grad():
            mesh = filter_degenerate_faces(dec_model(st)[0])

        v = mesh.vertices.detach().float()
        pts = subsample(v, args.n_sample, gen)
        masks = {th: render_mask(mesh, renderer, exts[th]) for th in ORBIT_ANGLES}
        gm = gt_mask(fi).to(DEVICE)

        if ref_pts is None:
            ref_pts = pts.clone()
            ref_masks = {k: m.clone() for k, m in masks.items()}
            ref_gt = gm.clone()
            # torch.cdist in float32 uses the expanded form and does not return
            # exactly 0 for a point set against itself. Measure that floor so a
            # real "the surface moved" signal is not confused with round-off.
            noise_floor = chamfer(ref_pts, ref_pts.clone())
            print(f'[NOISE] chamfer(frame1, frame1) = {noise_floor:.3e}  '
                  f'<- anything at or below this is numerical, not motion',
                  flush=True)

        cd = chamfer(pts, ref_pts) if fi != frames[0] else 0.0
        row = {
            'frame': fi,
            'n_verts': int(v.shape[0]), 'n_faces': int(mesh.faces.shape[0]),
            'chamfer_vs_f1': float(cd),
            'bbox': [float(x) for x in (v.max(0).values - v.min(0).values).tolist()],
            'sil_area':  {str(th): int(masks[th].sum()) for th in ORBIT_ANGLES},
            'sil_iou_f1': {str(th): float(iou(masks[th], ref_masks[th])) for th in ORBIT_ANGLES},
            'gt_area': int(gm.sum()), 'gt_iou_f1': float(iou(gm, ref_gt)),
        }
        rows.append(row)
        print(f'{fi:>6} {row["n_verts"]:>9,} {row["n_faces"]:>9,} '
              f'{cd:>14.6f} {row["sil_area"]["0.0"]:>14,} '
              f'{row["sil_iou_f1"]["0.0"]:>15.5f} {row["gt_area"]:>9,} '
              f'{row["gt_iou_f1"]:>8.5f}', flush=True)

        del st, mesh, v, pts, masks, gm
        gc.collect(); torch.cuda.empty_cache()

    # ── verdict ──────────────────────────────────────────────────────────────
    a_tr = np.array([r['sil_area']['0.0'] for r in rows], dtype=np.float64)
    a_gt = np.array([r['gt_area'] for r in rows], dtype=np.float64)
    d_tr = (a_tr[-1] - a_tr[0]) / a_tr[0] * 100
    d_gt = (a_gt[-1] - a_gt[0]) / a_gt[0] * 100
    rng_tr = (a_tr.max() - a_tr.min()) / a_tr.mean() * 100
    rng_gt = (a_gt.max() - a_gt.min()) / a_gt.mean() * 100
    iou_tr = min(r['sil_iou_f1']['0.0'] for r in rows)
    iou_gt = min(r['gt_iou_f1'] for r in rows)
    cd_max = max(r['chamfer_vs_f1'] for r in rows)
    snr    = cd_max / max(noise_floor, 1e-12)
    bbox0  = np.array(rows[0]['bbox'])
    cd_rel = cd_max / max(bbox0.max(), 1e-9) * 100

    print('\n' + '=' * 78)
    print('[VERDICT]')
    print('=' * 78)
    print(f'  TRELLIS frozen mesh:')
    print(f'    silhouette area, f1 -> f{frames[-1]} : {d_tr:+.2f}%')
    print(f'    silhouette area spread            : {rng_tr:.2f}% of mean')
    print(f'    lowest silhouette IoU vs frame 1  : {iou_tr:.5f}')
    print(f'    max chamfer to frame 1            : {cd_max:.6f} '
          f'({cd_rel:.3f}% of bbox)')
    print(f'    cdist noise floor                 : {noise_floor:.3e}  '
          f'-> signal is {snr:.1f}x the floor'
          + ('   <-- BELOW NOISE, surface did not move' if snr < 3 else ''))
    print(f'  GT video:')
    print(f'    silhouette area, f1 -> f{frames[-1]} : {d_gt:+.2f}%')
    print(f'    silhouette area spread            : {rng_gt:.2f}% of mean')
    print(f'    lowest silhouette IoU vs frame 1  : {iou_gt:.5f}')

    ratio = (rng_tr / rng_gt) if rng_gt > 1e-9 else float('nan')
    print(f'\n  TRACKING RATIO (TRELLIS spread / GT spread) = {ratio:.3f}')
    if rng_gt < 2.0:
        verdict = ('GT SHAPE IS ESSENTIALLY STATIC — this dataset cannot test '
                   'geometry dynamics. Use a video whose subject actually deforms.')
    elif ratio > 0.6:
        verdict = ('TRELLIS ALREADY TRACKS the shape change. A geometry LoRA is '
                   'solving a solved problem.')
    elif ratio < 0.2:
        verdict = ('SLaT CARRIES ALMOST NO SHAPE CHANGE while the GT deforms. '
                   'A geometry LoRA is justified.')
    else:
        verdict = 'PARTIAL tracking — a geometry LoRA would add, not replace.'
    print(f'  -> {verdict}')
    print('=' * 78, flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    fig.suptitle('Does the frozen decoder give time-varying geometry?', fontsize=11)
    f = [r['frame'] for r in rows]
    axes[0].plot(f, a_tr / a_tr[0], 'o-', color='#61afef', label='TRELLIS frozen')
    axes[0].plot(f, a_gt / a_gt[0], 's-', color='#e06c75', label='GT video')
    axes[0].set_xlabel('frame'); axes[0].set_ylabel('silhouette area / frame 1')
    axes[0].set_title(f'Silhouette area (ratio = {ratio:.2f})')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    for th, c in zip(ORBIT_ANGLES, ('#98c379', '#61afef', '#d19a66', '#c678dd')):
        axes[1].plot(f, [r['sil_iou_f1'][str(th)] for r in rows], 'o-',
                     color=c, ms=3, label=f'{th:.0f} deg')
    axes[1].plot(f, [r['gt_iou_f1'] for r in rows], 'k--', lw=2, label='GT (front)')
    axes[1].set_xlabel('frame'); axes[1].set_ylabel('silhouette IoU vs frame 1')
    axes[1].set_title('Shape drift, multi-view'); axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(f, [r['chamfer_vs_f1'] for r in rows], 'o-', color='#c678dd')
    axes[2].set_xlabel('frame'); axes[2].set_ylabel('chamfer distance to frame 1')
    axes[2].set_title(f'3D surface motion (bbox = {bbox0.max():.3f})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'geometry_change.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[SAVE] {OUT_DIR / "geometry_change.png"}')

    json.dump({'run_dir': str(RUN_DIR), 'gt_dir': str(GT_DIR),
               'frames': f, 'rows': rows,
               'trellis_area_change_pct': d_tr, 'gt_area_change_pct': d_gt,
               'trellis_area_spread_pct': rng_tr, 'gt_area_spread_pct': rng_gt,
               'min_iou_trellis': iou_tr, 'min_iou_gt': iou_gt,
               'max_chamfer': cd_max, 'chamfer_pct_bbox': cd_rel,
               'chamfer_noise_floor': noise_floor, 'chamfer_snr': snr,
               'tracking_ratio': ratio, 'verdict': verdict},
              open(OUT_DIR / 'geometry_change.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "geometry_change.json"}')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
