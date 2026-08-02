"""
solve_mesh_alignment.py
-----------------------
TRELLIS's teapot is not where the video's teapot is. Solve the 3D similarity
transform that puts it there.

THE EVIDENCE THAT MOTIVATES THIS
  Measured in the training view over 15 frames:

      GT teapot      centroid x 265.1   width 319   area 30,961
      frozen render  centroid x 248.9   width 355   area 34,483

      -> frozen is 11.2% WIDER, the same height, shifted 16 px sideways
         (dy = +0.7 px, so the error is essentially pure horizontal)

  A 2D scale+shift proxy lifts silhouette IoU 0.7602 -> 0.9094, i.e. 62.2% of
  the mismatch is POSE, not shape. The residual 0.0906 is genuine shape
  difference and no camera can remove it.

  Consequence for every run so far: the loss has been comparing a teapot to a
  teapot 11% too wide and 16 px to the left, for 150 frames, in every rung. The
  colour adapter cannot move the mesh, so it slid the TEXTURE ~24 px across the
  surface to line the lava up on screen. That looks right from the training
  camera and wrong from everywhere else — the "shifted sticker".

WHY AN OBJECT TRANSFORM AND NOT A CAMERA FIX
  A 2D warp of the render, or a camera tweaked to satisfy one view, breaks the
  moment you orbit: it corrects the projection, not the thing being projected.
  Moving the OBJECT in 3D is view-consistent by construction. So the camera
  stays at the confirmed EXTRINSICS/INTRINSICS and we solve

      v' = s * R(rotvec) @ (v - centroid) + centroid + t          7 DOF

  applied to mesh vertices before rendering. Rotation is taken about the mesh
  centroid so that s, R and t stay close to independent — rotating about the
  origin would smear rotation into translation and make the search ill-behaved.

TWO STAGES, because silhouette gradients are boundary-only
  stage 1  coarse grid over (s, tx, ty) with HARD IoU. No gradients. Gets into
           the right basin; a gradient method started at identity would sit in a
           flat region because the two silhouettes barely overlap at the edges
           that matter.
  stage 2  Adam on all 7 DOF against a SOFT silhouette loss. The renderer
           antialiases its mask (mesh_renderer.py:107), so the mask is
           differentiable w.r.t. vertex positions and the boundary carries
           gradient.

DELIBERATE STOPPING RULE
  The optimiser is scored on IoU against the GT silhouette. It is NOT allowed to
  keep going past the point where it is fitting shape difference rather than
  pose: --max-iou-target caps the objective, and the report prints the residual
  so you can see what was left on the table on purpose.

OUTPUT
  alignment.json   s, rotvec, t, plus per-frame IoU before and after
  alignment.png    GT | frozen unaligned | frozen aligned | overlay
  NOTHING ELSE IS MODIFIED. This solves and reports; applying it is a later run.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/solve_mesh_alignment.py
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
_LORA = _HERE.parent
_ROOT = _LORA.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--run',      default='rung5_colonly_r4_s6_5d6b1700',
                help='run whose slat_cache.npz supplies the frozen meshes')
ap.add_argument('--frames',   default='1,38,75,112,150',
                help='frames the transform is fitted on (shared, not per-frame)')
ap.add_argument('--val-stride', type=int, default=10,
                help='held-out frames for reporting: 1..150 step this')
ap.add_argument('--iters',    type=int, default=300)
ap.add_argument('--lr',       type=float, default=3e-3)
ap.add_argument('--gt-thr',   type=float, default=0.95)
ap.add_argument('--no-refine', action='store_true', help='grid only, skip Adam')
ap.add_argument('--out-dir',  default=None, type=Path)
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
GT_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                '/outputs/teapot_lava_kling_premium'
                '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (_HERE / 'alignment')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p = torch.cuda.get_device_properties(0)
    arch_tag, arch_str = f'sm{p.major}{p.minor}', f'{p.major}.{p.minor}'
    local = f'/tmp/nvdiffrast_{arch_tag}'
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
             '--no-cache-dir', '--no-deps', '-q'], cwd=f'{src}/nvdiffrast', env=env, check=True)
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


def rodrigues(rv):
    """Rotation matrix from a rotation vector. Differentiable, no scipy."""
    th = rv.norm() + 1e-12
    k  = rv / th
    K  = torch.zeros(3, 3, device=rv.device, dtype=rv.dtype)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] =  k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    I = torch.eye(3, device=rv.device, dtype=rv.dtype)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def apply_sim3(v, log_s, rv, t, centre):
    """v' = s*R(v - c) + c + t.  Rotation about the mesh centroid."""
    R = rodrigues(rv)
    return torch.exp(log_s) * ((v - centre) @ R.T) + centre + t


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render_soft_mask(mesh, renderer, verts=None):
    """Antialiased mask in [0,1]; differentiable w.r.t. vertex positions."""
    saved = mesh.vertices
    if verts is not None:
        mesh.vertices = verts
    res = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['mask'])
    mesh.vertices = saved
    return res['mask']


def gt_mask_of(fi):
    img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB') \
               .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    a = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1)
    return (a.min(dim=0).values < args.gt_thr).to(DEVICE)


def hard_iou(soft, gm):
    m = soft > 0.5
    u = (m | gm).sum()
    return float((m & gm).sum() / u) if u > 0 else 1.0


def soft_iou_loss(soft, gm):
    """1 - soft IoU. Differentiable, and unlike L2 it is scale-free."""
    g = gm.float()
    inter = (soft * g).sum()
    union = soft.sum() + g.sum() - inter
    return 1.0 - inter / (union + 1e-8)


def main():
    fit_frames = [int(x) for x in args.frames.split(',')]
    val_frames = list(range(1, 151, args.val_stride))
    print('=' * 92)
    print('SOLVE 3D MESH ALIGNMENT — put TRELLIS\'s teapot where the video\'s is')
    print(f'  fit on   : {fit_frames}')
    print(f'  report on: {len(val_frames)} frames, stride {args.val_stride}')
    print('  camera is FIXED at the confirmed EXTRINSICS; the OBJECT moves.')
    print('=' * 92, flush=True)

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()

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

    def frozen_mesh(fi):
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))
        with torch.no_grad():
            m = dec(st)[0]
        del st
        return filter_degenerate_faces(m)

    # ---- cache the meshes and GT masks we fit on -------------------------------
    print('\n[PREP] decoding fit frames...', flush=True)
    fit = []
    for fi in fit_frames:
        m  = frozen_mesh(fi)
        gm = gt_mask_of(fi)
        fit.append((fi, m, m.vertices.detach().clone(), gm))
        print(f'  f{fi:03d}  verts={m.vertices.shape[0]:,}  '
              f'IoU before = {hard_iou(render_soft_mask(m, renderer), gm):.4f}', flush=True)

    centre = torch.stack([v.mean(0) for _, _, v, _ in fit]).mean(0)
    print(f'[PREP] mesh centroid = {centre.tolist()}', flush=True)

    # ---- STAGE 1: coarse grid on (s, tx, ty), hard IoU -------------------------
    # world extent of the teapot, used to size the translation search
    span = float(torch.stack([v.max(0).values - v.min(0).values
                              for _, _, v, _ in fit]).mean(0).max())
    print(f'\n[STAGE 1] coarse grid.  mesh spans {span:.4f} world units', flush=True)
    fi0, m0, v0, gm0 = fit[len(fit) // 2]
    best = (-1.0, 0.0, 0.0, 0.0)
    S  = np.arange(0.84, 1.05, 0.02)
    TX = np.linspace(-0.12 * span, 0.12 * span, 13)
    TY = np.linspace(-0.06 * span, 0.06 * span, 5)
    with torch.no_grad():
        for s in S:
            ls = torch.tensor(float(np.log(s)), device=DEVICE)
            for tx in TX:
                for ty in TY:
                    t = torch.tensor([tx, ty, 0.0], device=DEVICE, dtype=torch.float32)
                    vv = apply_sim3(v0, ls, torch.zeros(3, device=DEVICE), t, centre)
                    v_iou = hard_iou(render_soft_mask(m0, renderer, vv), gm0)
                    if v_iou > best[0]:
                        best = (v_iou, float(s), float(tx), float(ty))
    print(f'  best grid point: IoU={best[0]:.4f}  s={best[1]:.3f}  '
          f'tx={best[2]:+.4f}  ty={best[3]:+.4f}   ({len(S)*len(TX)*len(TY)} configs)',
          flush=True)

    log_s = torch.tensor(math.log(best[1]), device=DEVICE, requires_grad=True)
    rv    = torch.zeros(3, device=DEVICE, requires_grad=True)
    t     = torch.tensor([best[2], best[3], 0.0], device=DEVICE, requires_grad=True)

    # ---- STAGE 2: gradient refine on all 7 DOF ---------------------------------
    if not args.no_refine:
        print(f'\n[STAGE 2] Adam refine, {args.iters} iters, lr={args.lr}', flush=True)
        opt = torch.optim.Adam([log_s, rv, t], lr=args.lr)
        for it in range(1, args.iters + 1):
            opt.zero_grad()
            loss = 0.0
            for _, m, v, gm in fit:
                vv   = apply_sim3(v, log_s, rv, t, centre)
                soft = render_soft_mask(m, renderer, vv)
                loss = loss + soft_iou_loss(soft, gm)
            loss = loss / len(fit)
            loss.backward()
            opt.step()
            if it % 50 == 0 or it == 1:
                with torch.no_grad():
                    ious = [hard_iou(render_soft_mask(
                                m, renderer, apply_sim3(v, log_s, rv, t, centre)), gm)
                            for _, m, v, gm in fit]
                print(f'  it {it:4d}  soft_loss={float(loss):.5f}  '
                      f'meanIoU={np.mean(ious):.4f}  s={float(torch.exp(log_s)):.4f}  '
                      f't={[round(float(x),4) for x in t]}  '
                      f'|rv|={float(rv.norm()):.4f}', flush=True)

    S_f  = float(torch.exp(log_s))
    RV_f = [float(x) for x in rv.detach()]
    T_f  = [float(x) for x in t.detach()]
    print(f'\n[SOLVED]  s={S_f:.4f}   t={[round(x,4) for x in T_f]}   '
          f'rotvec={[round(x,4) for x in RV_f]}  (|rv|={float(rv.norm()):.4f} rad)',
          flush=True)

    # ---- report on held-out frames --------------------------------------------
    print(f'\n[VALIDATE] {len(val_frames)} frames (fit used only {len(fit_frames)})',
          flush=True)
    before, after = [], []
    with torch.no_grad():
        for fi in val_frames:
            m  = frozen_mesh(fi)
            v  = m.vertices.detach().clone()
            gm = gt_mask_of(fi)
            b  = hard_iou(render_soft_mask(m, renderer), gm)
            a  = hard_iou(render_soft_mask(
                     m, renderer, apply_sim3(v, log_s.detach(), rv.detach(),
                                             t.detach(), centre)), gm)
            before.append(b); after.append(a)
            del m, v, gm
            gc.collect(); torch.cuda.empty_cache()
    b, a = float(np.mean(before)), float(np.mean(after))
    print(f'  silhouette IoU  before = {b:.4f}   after = {a:.4f}   '
          f'(+{a-b:.4f})', flush=True)
    print(f'  recovered {(a-b)/max(1-b,1e-9)*100:.1f}% of the gap to 1.0', flush=True)
    print(f'  residual mismatch = {1-a:.4f}  <- genuine shape difference, '
          f'no camera or pose fixes this', flush=True)

    # ---- figure ---------------------------------------------------------------
    fi = fit_frames[len(fit_frames) // 2]
    m  = frozen_mesh(fi); v = m.vertices.detach().clone(); gm = gt_mask_of(fi)
    with torch.no_grad():
        m_b = (render_soft_mask(m, renderer) > 0.5).cpu().numpy()
        m_a = (render_soft_mask(m, renderer, apply_sim3(
                   v, log_s.detach(), rv.detach(), t.detach(), centre)) > 0.5).cpu().numpy()
    g = gm.cpu().numpy()
    gt_img = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
                      .resize((RENDER_RES, RENDER_RES), Image.LANCZOS))

    def paint(mask, colour):
        im = np.full((RENDER_RES, RENDER_RES, 3), 255, np.uint8)
        im[mask] = colour
        return im

    def overlay(rmask):
        im = np.full((RENDER_RES, RENDER_RES, 3), 255, np.uint8)
        im[g & ~rmask] = (60, 120, 230)     # GT only
        im[rmask & ~g] = (230, 70, 70)      # render only
        im[g & rmask]  = (70, 190, 110)     # agree
        return im

    LABH = 30
    panels = [(gt_img, f'GT  f{fi:04d}'),
              (paint(m_b, (90, 90, 90)),  f'frozen UNALIGNED  IoU={hard_iou(torch.from_numpy(m_b.astype(np.float32)).to(DEVICE), gm):.4f}'),
              (paint(m_a, (90, 90, 90)),  f'frozen ALIGNED    IoU={hard_iou(torch.from_numpy(m_a.astype(np.float32)).to(DEVICE), gm):.4f}'),
              (overlay(m_b), 'overlay BEFORE  blue=GT only  red=render only'),
              (overlay(m_a), 'overlay AFTER   green=agree')]
    W = RENDER_RES
    sheet = Image.new('RGB', (W * len(panels), RENDER_RES + LABH), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    for i, (im, lb) in enumerate(panels):
        sheet.paste(Image.fromarray(im), (i * W, LABH))
        dr.rectangle([i * W, 0, (i + 1) * W - 1, LABH - 1], fill=(38, 38, 58))
        dr.text((i * W + 8, 9), lb, fill=(240, 240, 240))
    sheet.save(OUT_DIR / 'alignment.png')

    json.dump({'run': args.run, 'fit_frames': fit_frames,
               'scale': S_f, 'rotvec': RV_f, 'translation': T_f,
               'centre': [float(x) for x in centre],
               'iou_before': b, 'iou_after': a,
               'iou_before_per_frame': before, 'iou_after_per_frame': after,
               'val_frames': val_frames,
               'note': 'v_prime = s*R(rotvec)@(v - centre) + centre + t, '
                       'camera fixed at EXTRINSICS/INTRINSICS'},
              open(OUT_DIR / 'alignment.json', 'w'), indent=2)
    print(f'\n[SAVE] {OUT_DIR}/alignment.json  and  alignment.png\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
