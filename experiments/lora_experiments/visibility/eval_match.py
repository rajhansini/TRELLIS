"""
eval_match.py
-------------
How well does the rung's render actually match the video, per frame, measured
through the REAL rasterizer?

WHY NOT JUST READ THE TRAINING LOG
  Each rung's held-out PSNR/SSIM/LPIPS is computed on rung15's UNION mask
  ((render|gt), gt_mask = (gt<0.99).any()), which covers 99.8% of the frame.
  That number is (a) dominated by background and (b) not comparable across rungs
  that trained on different regions. It cannot answer "are the renders matching".

  Everything here is computed on the INTERSECTION of the rendered silhouette and
  a tight GT foreground mask, so only pixels that are teapot in BOTH count.

WHY NOT SPLAT THE VERTICES
  Building the silhouette by projecting vertices undercounts area ~12% and
  silently biases every downstream number (it previously produced a fake 'the
  alignment scale is wrong' result). The mask here comes from MeshRenderer --
  the same rasterizer the training loss uses.

  The mesh is aligned exactly ONCE, inside this script, from the raw decoder
  output. export_xattn_mesh.py aligns by default; feeding one of its PLYs into
  an aligner again costs 0.93 -> 0.75 IoU for that reason alone.

METRICS, per frame, frozen and adapted
  sil_IoU        rendered silhouette vs tight GT foreground. Geometry/pose.
  PSNR, MSE      on the intersection, 3-channel. Appearance where both agree.
  bright_IoU     {min(RGB) > --bright} in render vs the same in GT. This is the
                 white-patch metric: it separates "how much is bright" (counts)
                 from "is it in the right place" (IoU).
  chance         GT bright fraction of the silhouette -- the IoU-precision a
                 random assignment would reach. bright numbers are meaningless
                 without it.
  sat_p80/p90    fraction of rendered object pixels above 0.8 / 0.9. Direct
                 saturation readout; rung15 saturated, rung19 should not.

Usage:
  python .../eval_match.py --run rung19_... --frames 1 5 15 ... 145
"""

import sys, os, argparse, json, gc
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LEX  = _HERE.parent
_ROOT = _LEX.parent.parent

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--run', default='rung19_intersection_uniform_all_qkvo_r4_s6_276b5db7')
_pre.add_argument('--ckpt', default='lora_best.pt')
_pre.add_argument('--frames', type=int, nargs='+',
                  default=[1] + list(range(5, 151, 10)))
_pre.add_argument('--bright', type=float, default=0.5)
_pre.add_argument('--out', default=None)
A, _rest = _pre.parse_known_args()

OUT = Path(A.out) if A.out else (_HERE / 'eval_match')
OUT.mkdir(parents=True, exist_ok=True)

# export_xattn_mesh.py owns the verified flow/LoRA/align code. Import it rather
# than re-implement, with argv set so its module-level argparse succeeds.
sys.argv = ['export_xattn_mesh.py', '--run', A.run, '--ckpt', A.ckpt,
            '--frame', '1', '--no-glb', '--out-dir', str(OUT / '_scratch')]
sys.path.insert(0, str(_ROOT / 'render'))
import export_xattn_mesh as X                                    # noqa: E402

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from PIL import Image                                            # noqa: E402

sys.path.insert(0, str(_ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))
from trellis.representations.mesh import MeshExtractResult       # noqa: E402
from step8_decode_render.decode_render import (                  # noqa: E402
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = X.DEVICE
GTD = X.GT_FRAMES_DIR


class _Tee:
    def __init__(self, p): self._f = open(p, 'a', buffering=1)
    def write(self, m): sys.__stdout__.write(m); self._f.write(m)
    def flush(self): sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(OUT / 'log.txt'); sys.stderr = sys.stdout


def gt_of(i):
    im = Image.open(GTD / f'frame_{i:04d}.png').convert('RGB') \
              .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    g = torch.from_numpy(np.asarray(im, np.float32) / 255.).to(DEVICE)
    return g, (g.min(dim=2).values < 0.95)


def render_mesh(mesh, renderer):
    v = mesh.vertices.detach()
    f = mesh.faces.detach()
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    f = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
    v = X.align(v)                                   # exactly once, from raw
    col = mesh.vertex_attrs[:, :3].detach().clamp(0, 1)
    mr = MeshExtractResult(vertices=v, faces=f,
                           vertex_attrs=torch.cat([col, col], dim=1), res=256)
    r = renderer.render(mr, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                        return_types=['color', 'mask'])
    c = r['color'].squeeze()
    c = c.permute(1, 2, 0) if c.shape[0] == 3 else c
    return c.clamp(0, 1), (r['mask'].squeeze() > 0.5)


def iou(a, b):
    i = (a & b).sum().item(); u = (a | b).sum().item()
    return i / max(u, 1)


def main():
    print('=' * 96)
    print(f'EVAL MATCH  run={A.run}  ckpt={A.ckpt}  bright>{A.bright}')
    print('=' * 96, flush=True)

    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained(X.PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(GTD / 'frame_0075.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(X.STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    assert coords.shape[0] == 7301, f'N_vox={coords.shape[0]} != 7301'
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass
    dino = pipe.models['image_cond_model'].to(DEVICE)

    rd = _LEX / 'runs' / A.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / A.ckpt, map_location='cpu', weights_only=True)
    reg = X.XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                              tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    bmax = max(float(getattr(b, f'lora_{n}').B.float().norm())
               for b in reg.blocks.values() for n in b.targets)
    assert bmax > 1e-6, 'LoRA B all zero -- adapter not loaded'
    print(f'[LORA] epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}  '
          f'max||B|| {bmax:.4f}', flush=True)

    renderer = make_renderer(DEVICE)
    torch.manual_seed(X.NOISE_SEED)
    noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)

    hdr = (f'\n{"fr":>4s} {"tag":7s} {"silIoU":>7s} {"PSNR":>7s} {"brRnd":>6s} '
           f'{"brGT":>6s} {"brIoU":>6s} {"chance":>7s} {"p80%":>6s} {"p90%":>6s}')
    print(hdr); print('-' * 78, flush=True)

    rows = []
    for fi in A.frames:
        g, gfg = gt_of(fi)
        gbr = gfg & (g.min(dim=2).values > A.bright)
        cond = X.encode(dino, fi).unsqueeze(0).to(DEVICE)
        for tag, L in (('frozen', None), ('rung', reg)):
            X.LORA = L
            slat = X.denoise(fm, noise, coords, cond)
            with torch.no_grad():
                mesh = dec(slat)[0]
            col, sil = render_mesh(mesh, renderer)
            inter = sil & gfg
            n = int(inter.sum())
            mse = ((col - g) ** 2)[inter].mean().item() if n else float('nan')
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            mn = col.min(dim=2).values
            rbr = sil & (mn > A.bright)
            chance = int(gbr.sum()) / max(int(sil.sum()), 1)
            row = dict(frame=fi, tag=tag, sil_iou=iou(sil, gfg), psnr=psnr,
                       br_rnd=int(rbr.sum()), br_gt=int(gbr.sum()),
                       br_iou=iou(rbr, gbr), chance=chance,
                       p80=float((sil & (mn > 0.8)).sum() / max(int(sil.sum()), 1)),
                       p90=float((sil & (mn > 0.9)).sum() / max(int(sil.sum()), 1)))
            rows.append(row)
            print(f'{fi:4d} {tag:7s} {row["sil_iou"]:7.4f} {psnr:7.3f} '
                  f'{row["br_rnd"]:6d} {row["br_gt"]:6d} {row["br_iou"]:6.4f} '
                  f'{chance:7.4f} {100*row["p80"]:6.2f} {100*row["p90"]:6.2f}',
                  flush=True)
            del slat, mesh, col, sil
            gc.collect(); torch.cuda.empty_cache()

    print('\n' + '=' * 96)
    print('MEANS')
    for tag in ('frozen', 'rung'):
        r = [x for x in rows if x['tag'] == tag]
        f = lambda k: np.mean([x[k] for x in r])
        print(f'  {tag:7s} silIoU {f("sil_iou"):.4f}   PSNR {f("psnr"):6.3f}   '
              f'brightIoU {f("br_iou"):.4f} (chance {f("chance"):.4f}, '
              f'{f("br_iou")/max(f("chance"),1e-9):.1f}x)   '
              f'bright {f("br_rnd"):6.0f} px vs GT {f("br_gt"):6.0f} px   '
              f'>0.8 {100*f("p80"):.2f}%  >0.9 {100*f("p90"):.2f}%')
    json.dump(rows, open(OUT / 'eval_match.json', 'w'), indent=1)
    print(f'\n[DONE] {OUT}/eval_match.json')


if __name__ == '__main__':
    main()
