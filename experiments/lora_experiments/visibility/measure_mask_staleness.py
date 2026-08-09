"""
measure_mask_staleness.py
-------------------------
How wrong is frame 75's frozen silhouette as a loss stencil for the other 149?

WHAT THE TRAINING ACTUALLY DOES  (rung19/rung20, traced line by line)
  coords  = sample_sparse_structure(frame 75)      FIXED, 7,301 voxels, all frames
  noise   = randn(seed 6)                          FIXED
  cond_i  = DINO(frame i)                          VARIES
  slat_i  = denoise(noise, coords, cond_i)         VARIES  -> 150 different latents
  mesh_i  = slat_decoder_mesh(slat_i)              VARIES  -> 150 different meshes
  render_mask = frozen_reference(cond_75)['sil']   FIXED   <- ONE silhouette

  so the loss region for EVERY frame is
      m_i = sil(frozen mesh @ frame 75)  &  tight_gt(frame i)
  when the honest region would be
      m_i = sil(adapted mesh @ frame i)  &  tight_gt(frame i)

WHAT IS MEASURED
  For each probed frame, render the FROZEN mesh for THAT frame and compare its
  silhouette against frame 75's, which is the stencil actually in use:
    - IoU(sil_i, sil_75)          1.0 would mean the stencil is exact
    - px in sil_i but NOT in the stencil   -> real render never scored
    - px in the stencil but NOT in sil_i   -> background scored as if object
  Both errors are one-sided and neither is visible in any metric we report.

  The noise floor matters: the pipeline is not bit-reproducible, so frame 75 is
  run TWICE and IoU(run1, run2) is reported first. Any frame-to-frame IoU above
  that floor is real variation; anything at it is sampling noise.

  Also reports the same against the tight GT mask per frame, so the drift can be
  compared against the thing the stencil is intersected with.
"""
import sys, os, argparse, gc
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _LEX = _HERE.parent; _ROOT = _LEX.parent.parent
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--frames', type=int, nargs='+',
                  default=[1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
_pre.add_argument('--run', default='rung20_intersection_uniform_all_kv_r4_s6_c7a82c89')
A, _ = _pre.parse_known_args()
OUT = _HERE / 'mask_staleness'; OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_xattn_mesh.py', '--run', A.run, '--frame', '75',
            '--no-glb', '--out-dir', str(OUT/'_scratch')]
sys.path.insert(0, str(_ROOT/'render'))
import export_xattn_mesh as X
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(_ROOT/'experiments'/'dynamic_texture_trellis_pipeline'))
from trellis.representations.mesh import MeshExtractResult
from step8_decode_render.decode_render import (make_renderer, RENDER_RES,
                                               EXTRINSICS, INTRINSICS)
DEVICE = X.DEVICE

from trellis.pipelines import TrellisImageTo3DPipeline
pipe = TrellisImageTo3DPipeline.from_pretrained(X.PRETRAINED); pipe.to(DEVICE)
fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
for p in fm.parameters(): p.requires_grad_(False)
for p in dec.parameters(): p.requires_grad_(False)
ref = Image.open(X.GT_FRAMES_DIR/'frame_0075.png').convert('RGB')
cs = pipe.get_cond([ref]); torch.manual_seed(X.STRUCT_SEED)
coords = pipe.sample_sparse_structure(cs, num_samples=1)
assert coords.shape[0] == 7301
del cs; gc.collect(); torch.cuda.empty_cache()
for n in list(pipe.models):
    if n not in {'slat_flow_model','slat_decoder_mesh','image_cond_model'}:
        try: pipe.models[n].cpu()
        except Exception: pass
dino = pipe.models['image_cond_model'].to(DEVICE)
r = make_renderer(DEVICE)
torch.manual_seed(X.NOISE_SEED)
noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)
X.LORA = None

def sil_of(frame):
    cond = X.encode(dino, frame).unsqueeze(0).to(DEVICE)
    slat = X.denoise(fm, noise, coords, cond)
    with torch.no_grad(): mesh = dec(slat)[0]
    v = X.align(mesh.vertices.detach()); f = mesh.faces.detach()
    e1,e2 = v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]]
    f = f[0.5*torch.cross(e1,e2,dim=1).norm(dim=1) > 1e-6]
    c = mesh.vertex_attrs[:, :3].detach().clamp(0,1)
    res = r.render(MeshExtractResult(vertices=v, faces=f,
                   vertex_attrs=torch.cat([c,c],1), res=256),
                   EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE), return_types=['mask'])
    s = (res['mask'].squeeze() > 0.5).clone()
    nv = int(mesh.vertices.shape[0])
    del slat, mesh, res; gc.collect(); torch.cuda.empty_cache()
    return s, nv

def gt_of(i):
    im = Image.open(X.GT_FRAMES_DIR/f'frame_{i:04d}.png').convert('RGB') \
              .resize((RENDER_RES,RENDER_RES), Image.LANCZOS)
    g = torch.from_numpy(np.asarray(im, np.float32)/255.).to(DEVICE)
    return g.min(dim=2).values < 0.95

def iou(a,b):
    return int((a&b).sum())/max(int((a|b).sum()),1)

print('='*94)
print('MASK STALENESS  -- frame 75 frozen silhouette used as the stencil for all 150')
print('='*94, flush=True)
s75a, v75a = sil_of(75)
s75b, v75b = sil_of(75)
floor = iou(s75a, s75b)
print(f'[NOISE FLOOR] frame 75 run twice: IoU {floor:.4f}   '
      f'verts {v75a:,} vs {v75b:,} (delta {abs(v75a-v75b)})')
print('  any frame-to-frame IoU BELOW this is real variation, not sampling noise\n')

STEN = s75a
print(f'{"fr":>4s} {"verts":>9s} {"sil px":>8s} {"IoU vs stencil":>15s} '
      f'{"render NOT scored":>18s} {"stencil off-object":>19s} {"IoU(sil,GT)":>12s}')
print('-'*94)
rows=[]
for fi in A.frames:
    s, nv = sil_of(fi)
    g = gt_of(fi)
    miss = int((s & ~STEN).sum())      # real render outside the stencil
    extra = int((STEN & ~s).sum())     # stencil covering where this mesh is not
    j = iou(s, STEN)
    rows.append((fi, nv, int(s.sum()), j, miss, extra, iou(s,g)))
    print(f'{fi:4d} {nv:9,d} {int(s.sum()):8,d} {j:15.4f} {miss:18,d} '
          f'{extra:19,d} {iou(s,g):12.4f}')

J = np.array([x[3] for x in rows]); M = np.array([x[4] for x in rows])
E = np.array([x[5] for x in rows]); V = np.array([x[1] for x in rows])
print('-'*94)
print(f'IoU vs stencil : min {J.min():.4f}  mean {J.mean():.4f}  max {J.max():.4f}'
      f'   (noise floor {floor:.4f})')
print(f'verts          : min {V.min():,}  max {V.max():,}  spread {V.max()-V.min():,} '
      f'({100*(V.max()-V.min())/V.mean():.2f}%)')
print(f'render NOT scored   : mean {M.mean():,.0f} px  max {M.max():,} px')
print(f'stencil off-object  : mean {E.mean():,.0f} px  max {E.max():,} px')
print(f'\nVERDICT: mean IoU {J.mean():.4f} vs noise floor {floor:.4f} -> '
      f'{"REAL, the stencil is stale" if J.mean() < floor - 0.005 else "within noise, stencil is fine"}')
np.savez_compressed(OUT/'staleness.npz', rows=np.array(rows), floor=floor)
print(f'[DONE] {OUT}/staleness.npz')
