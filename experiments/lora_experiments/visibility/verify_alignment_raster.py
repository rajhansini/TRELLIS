"""
verify_alignment_raster.py
--------------------------
Settle whether alignment.json is mis-solved, using the REAL rasterizer.

WHY THIS EXISTS
  A CPU analysis of rung19's frame-1 mesh built the render silhouette by
  SPLATTING vertices. That undercounts area badly -- 25,351 px raw, 26,488 px
  after closing+fill, against GT's 30,007 px, i.e. ~12% short. Fitting a 7-DOF
  similarity against an under-filled mask pushes the SCALE up to recover the
  missing area, and it duly "found" scale 0.890 -> 0.985 with IoU 0.768 -> 0.904.

  That conclusion is unsafe. alignment.json records iou_after 0.9051 (frame 1:
  0.9180) measured with nvdiffrast at solve time, versus the 0.7681 the splat
  reports for the SAME transform. Splat area deficit fully accounts for the gap.

  So the question is decided by rasterizing, not splatting. This renders the
  mask through MeshRenderer -- the same renderer the loss uses -- and reports
  IoU with the alignment ON and OFF.

WHAT IT DECIDES
  IoU(aligned) ~= 0.90 on frame 1
      -> alignment.json is correct, the splat analysis was an artefact, and the
         residual bright-patch misplacement is NOT an alignment problem.
  IoU(aligned) ~= 0.77
      -> alignment.json really does underperform its own recorded value and is
         worth re-solving.

  It also re-fits scale ALONE against the rasterized mask (1-D, so it cannot be
  confounded the way the 7-DOF splat fit was) and reports the IoU-optimal scale.
  If that lands near 0.890 the current value is right.

  Finally it reports bright-region IoU from the rasterized colour buffer, which
  is the number the splat put at 0.0135 and which motivated everything above.

Usage:
  python .../verify_alignment_raster.py --frame 1
"""

import sys, os, argparse, json, math
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_LEX  = _HERE.parent
_ROOT = _LEX.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--frame', type=int, default=1)
ap.add_argument('--mesh', default=str(_ROOT / 'render/out/rung19_i_f0001/result_f0001.ply'))
ap.add_argument('--bright', type=float, default=0.5)
ap.add_argument('--already-aligned', action='store_true',
                help='export_xattn_mesh.py applies the alignment by default, so a '
                     'PLY exported WITHOUT --no-align is already aligned. Set this '
                     'to undo it first, otherwise the transform is applied twice '
                     'and IoU drops 0.9286 -> 0.7496 for that reason alone.')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'align_verify')
OUT.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, p): self._f = open(p, 'a', buffering=1)
    def write(self, m): sys.__stdout__.write(m); self._f.write(m)
    def flush(self): sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(OUT / 'log.txt'); sys.stderr = sys.stdout

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_PIPE))
from trellis.representations.mesh import MeshExtractResult
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
ALIGN = json.load(open(_HERE / 'alignment' / 'alignment.json'))


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k = np.asarray(rv, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


_AR = torch.tensor(rodrigues(ALIGN['rotvec']), dtype=torch.float32, device=DEVICE)
_AC = torch.tensor(ALIGN['centre'], dtype=torch.float32, device=DEVICE)
_AT = torch.tensor(ALIGN['translation'], dtype=torch.float32, device=DEVICE)


def apply_align(v, scale):
    return scale * ((v - _AC) @ _AR.T) + _AC + _AT


def unapply_align(v, scale):
    """Exact inverse of apply_align, so a PLY that was exported already-aligned
    can be returned to raw mesh space and the scale sweep stays meaningful."""
    return ((v - _AT - _AC) @ _AR) / scale + _AC


def render(verts, faces, cols, renderer):
    mesh = MeshExtractResult(vertices=verts, faces=faces,
                             vertex_attrs=torch.cat([cols, cols], dim=1), res=256)
    r = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                        return_types=['color', 'mask'])
    return r['color'], r['mask']


def iou(a, b):
    i = (a & b).sum().item(); u = (a | b).sum().item()
    return i / max(u, 1)


def main():
    im = Image.open(GT_FRAMES_DIR / f'frame_{args.frame:04d}.png').convert('RGB') \
              .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    g = torch.from_numpy(np.asarray(im, np.float32) / 255.).to(DEVICE)
    gt_fg = g.min(dim=2).values < 0.95
    gt_br = gt_fg & (g.min(dim=2).values > args.bright)
    print(f'GT frame {args.frame}   fg {int(gt_fg.sum()):,} px   '
          f'bright {int(gt_br.sum()):,} px')

    m = trimesh.load(args.mesh, process=False)
    V = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEVICE)
    F = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=DEVICE)
    C = torch.tensor(np.asarray(m.visual.vertex_colors)[:, :3] / 255.,
                     dtype=torch.float32, device=DEVICE)
    print(f'mesh {args.mesh}\n     {len(V):,} verts  {len(F):,} faces')

    renderer = make_renderer(DEVICE)
    s_cur = float(ALIGN['scale'])

    # GATE-double: a PLY from export_xattn_mesh.py without --no-align is ALREADY
    # aligned. Undo it here or every number below is the transform applied twice.
    V_raw = unapply_align(V, s_cur) if args.already_aligned else V
    if args.already_aligned:
        rt = apply_align(V_raw, s_cur)
        err = (rt - V).abs().max().item()
        print(f'[GATE-double] mesh was already aligned; undone. '
              f'round-trip max err {err:.3e}')
        assert err < 1e-4, f'inverse is not exact: {err}'

    print('\n' + '=' * 62)
    print('1. RASTERIZED IoU  (the number the splat got wrong)')
    for tag, sc in (('raw mesh      ', None), (f'aligned s={s_cur:.5f}', s_cur)):
        v = V_raw if sc is None else apply_align(V_raw, sc)
        col, msk = render(v, F, C, renderer)
        sil = msk.squeeze() > 0.5
        print(f'   {tag}   render {int(sil.sum()):6,d} px   IoU {iou(sil, gt_fg):.4f}')
    print(f'   alignment.json recorded  iou_before {ALIGN["iou_before"]:.4f}'
          f'   iou_after {ALIGN["iou_after"]:.4f}')
    print(f'   frame-1 recorded after   {ALIGN["iou_after_per_frame"][0]:.4f}')

    print('\n2. SCALE SWEEP against the RASTERIZED mask (1-D, unconfounded)')
    best = (-1, None)
    for sc in np.arange(0.84, 1.06, 0.01):
        v = apply_align(V_raw, float(sc))
        _, msk = render(v, F, C, renderer)
        j = iou(msk.squeeze() > 0.5, gt_fg)
        flag = '  <- current' if abs(sc - s_cur) < 0.005 else ''
        print(f'   s {sc:.3f}   IoU {j:.4f}{flag}')
        if j > best[0]: best = (j, float(sc))
    print(f'   IoU-optimal scale {best[1]:.3f}  (IoU {best[0]:.4f})   '
          f'current {s_cur:.5f}')

    print('\n3. BRIGHT-REGION IoU from the rasterized colour buffer')
    v = apply_align(V_raw, s_cur)
    col, msk = render(v, F, C, renderer)
    sil = msk.squeeze() > 0.5
    cc = col.squeeze().permute(1, 2, 0) if col.squeeze().shape[0] == 3 else col.squeeze()
    br = sil & (cc.min(dim=2).values > args.bright)
    print(f'   render bright {int(br.sum()):,} px   GT bright {int(gt_br.sum()):,} px')
    print(f'   bright IoU {iou(br, gt_br):.4f}   (splat claimed 0.0135)')
    print('=' * 62)


if __name__ == '__main__':
    main()
