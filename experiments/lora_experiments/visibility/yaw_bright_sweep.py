"""
yaw_bright_sweep.py
-------------------
Are the white patches seen in the ORBIT video on surfaces that training never saw?

Training renders one camera only (yaw=0, the confirmed EXTRINSICS). Every metric
reported so far was measured at that same yaw, so it is blind to anything the
adapter did to surfaces facing away. Two facts motivate the check:
  - 7.10% of rung19's VERTICES are bright (min-ch > 0.5) but only 2.87% of the
    pixels VISIBLE at yaw=0 are. The difference has to be somewhere.
  - the LoRA acts on the 32^3 cross-attention token grid; one token voxel spans
    ~8 decoder cubes per axis, covering front AND back surface simultaneously.
    Brightening a supervised front voxel necessarily brightens its unsupervised
    back side.

So: render the SAME already-aligned mesh from many yaws and count bright pixels.
Flat across yaw  -> the front-view metrics are representative.
Rising away from 0 -> the white lives on unsupervised surface, and no amount of
silhouette matching at yaw=0 will remove it.

The mesh is loaded from an export that is ALREADY aligned, so no alignment is
applied here -- doing so would double it (costs 0.93 -> 0.75 IoU on its own).
"""
import sys, os, argparse, math
from pathlib import Path
os.environ['SPCONV_ALGO'] = 'native'; os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
_HERE = Path(__file__).resolve().parent; _ROOT = _HERE.parent.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
ap = argparse.ArgumentParser()
ap.add_argument('--mesh', default=str(_ROOT/'render/out/rung19_i_f0001/result_f0001.ply'))
ap.add_argument('--frozen', default=str(_ROOT/'render/out/rung19_i_f0001/frozen_f0001.ply'))
ap.add_argument('--yaws', type=float, nargs='+',
                default=[0,30,60,90,120,150,180,210,240,270,300,330])
ap.add_argument('--bright', type=float, default=0.5)
args = ap.parse_args()
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_PIPE))
import numpy as np, torch, trimesh
from trellis.representations.mesh import MeshExtractResult
from step8_decode_render.decode_render import make_renderer, RENDER_RES, INTRINSICS
DEVICE = torch.device('cuda')

def orbit_extrinsics(yaw_deg, elev_deg=0.0, radius=2.0):
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius*math.cos(e)*math.sin(y), -radius*math.cos(e)*math.cos(y),
                    radius*math.sin(e)])
    fwd = -eye/np.linalg.norm(eye); up_w = np.array([0.,0.,1.])
    right = np.cross(fwd, up_w); right /= np.linalg.norm(right)+1e-12
    up = np.cross(right, fwd)
    ext = np.eye(4); ext[0,:3], ext[1,:3], ext[2,:3] = right, -up, fwd
    ext[:3,3] = -ext[:3,:3] @ eye
    return torch.tensor(ext, dtype=torch.float32)

def load(p):
    m = trimesh.load(p, process=False)
    V = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEVICE)
    F = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=DEVICE)
    C = torch.tensor(np.asarray(m.visual.vertex_colors)[:,:3]/255.,
                     dtype=torch.float32, device=DEVICE)
    return V, F, C

r = make_renderer(DEVICE)
# GATE-cam: yaw=0 must reproduce the confirmed training EXTRINSICS.
from step8_decode_render.decode_render import EXTRINSICS
d = (orbit_extrinsics(0.0).to(DEVICE) - EXTRINSICS.to(DEVICE)).abs().max().item()
print(f'[GATE-cam] yaw=0 vs training EXTRINSICS max diff {d:.2e}')
assert d < 1e-5, 'orbit camera does not reproduce the training view'

print(f'\n{"yaw":>5s} | {"rung19: obj px":>14s} {">0.5":>7s} {">0.8":>7s} {">0.9":>7s}'
      f' | {"frozen: >0.5":>13s}')
print('-'*70)
rows=[]
for tag, path in (('rung19', args.mesh), ('frozen', args.frozen)):
    V,F,C = load(path); out=[]
    for y in args.yaws:
        res = r.render(MeshExtractResult(vertices=V, faces=F,
                        vertex_attrs=torch.cat([C,C],1), res=256),
                       orbit_extrinsics(y).to(DEVICE), INTRINSICS.to(DEVICE),
                       return_types=['color','mask'])
        c = res['color'].squeeze(); c = c.permute(1,2,0) if c.shape[0]==3 else c
        m = res['mask'].squeeze() > 0.5; mn = c.clamp(0,1).min(dim=2).values
        n = max(int(m.sum()),1)
        out.append((int(m.sum()), 100*float((m&(mn>0.5)).sum())/n,
                    100*float((m&(mn>0.8)).sum())/n, 100*float((m&(mn>0.9)).sum())/n))
    rows.append((tag,out))
for i,y in enumerate(args.yaws):
    a=rows[0][1][i]; b=rows[1][1][i]
    print(f'{y:5.0f} | {a[0]:14,d} {a[1]:6.2f}% {a[2]:6.2f}% {a[3]:6.2f}% | {b[1]:12.2f}%')
a=[x[1] for x in rows[0][1]]
print(f'\nrung19 >0.5 across yaw:  min {min(a):.2f}%  max {max(a):.2f}%  '
      f'at yaw=0 {a[0]:.2f}%   max/yaw0 = {max(a)/max(a[0],1e-9):.2f}x')
