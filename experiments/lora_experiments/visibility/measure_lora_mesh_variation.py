"""Do rung20's ADAPTED meshes differ frame to frame? Vertices, faces, silhouette.
Frozen is measured alongside as the reference, and frame 75 is run twice to give
the non-determinism floor -- without it a delta of a few hundred verts is
unreadable."""
import sys, os, argparse, json, gc
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _LEX = _HERE.parent; _ROOT = _LEX.parent.parent
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--frames', type=int, nargs='+', default=[1,25,50,75,100,125,150])
_pre.add_argument('--run', default='rung20_intersection_uniform_all_kv_r4_s6_c7a82c89')
A,_ = _pre.parse_known_args()
OUT = _HERE/'lora_mesh_var'; OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_xattn_mesh.py','--run',A.run,'--frame','75','--no-glb','--out-dir',str(OUT/'_s')]
sys.path.insert(0, str(_ROOT/'render'))
import export_xattn_mesh as X
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(_ROOT/'experiments'/'dynamic_texture_trellis_pipeline'))
from trellis.representations.mesh import MeshExtractResult
from step8_decode_render.decode_render import make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS
DEVICE = X.DEVICE
from trellis.pipelines import TrellisImageTo3DPipeline
pipe = TrellisImageTo3DPipeline.from_pretrained(X.PRETRAINED); pipe.to(DEVICE)
fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
for p in fm.parameters(): p.requires_grad_(False)
for p in dec.parameters(): p.requires_grad_(False)
ref = Image.open(X.GT_FRAMES_DIR/'frame_0075.png').convert('RGB')
cs = pipe.get_cond([ref]); torch.manual_seed(X.STRUCT_SEED)
coords = pipe.sample_sparse_structure(cs, num_samples=1); assert coords.shape[0]==7301
del cs; gc.collect(); torch.cuda.empty_cache()
for n in list(pipe.models):
    if n not in {'slat_flow_model','slat_decoder_mesh','image_cond_model'}:
        try: pipe.models[n].cpu()
        except Exception: pass
dino = pipe.models['image_cond_model'].to(DEVICE)
rd = _LEX/'runs'/A.run; cfg = json.load(open(rd/'config.json'))
ck = torch.load(rd/'lora_ckpts/lora_best.pt', map_location='cpu', weights_only=True)
reg = X.XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'], tuple(cfg['targets'])).to(DEVICE)
reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
print(f"[LORA] epoch {ck['epoch']} psnr {ck['best_psnr']:.3f} targets {tuple(cfg['targets'])}")
r = make_renderer(DEVICE)
torch.manual_seed(X.NOISE_SEED)
noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)

def mesh_of(frame, L):
    X.LORA = L
    cond = X.encode(dino, frame).unsqueeze(0).to(DEVICE)
    slat = X.denoise(fm, noise, coords, cond)
    with torch.no_grad(): m = dec(slat)[0]
    v = X.align(m.vertices.detach()); f = m.faces.detach()
    e1,e2 = v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]]
    f2 = f[0.5*torch.cross(e1,e2,dim=1).norm(dim=1) > 1e-6]
    c = m.vertex_attrs[:,:3].detach().clamp(0,1)
    res = r.render(MeshExtractResult(vertices=v, faces=f2, vertex_attrs=torch.cat([c,c],1), res=256),
                   EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE), return_types=['mask'])
    s = (res['mask'].squeeze()>0.5).clone()
    nv, nf = int(m.vertices.shape[0]), int(f.shape[0])
    del slat, m, res; gc.collect(); torch.cuda.empty_cache()
    return nv, nf, s

def iou(a,b): return int((a&b).sum())/max(int((a|b).sum()),1)

na,fa,sa = mesh_of(75, reg); nb,fb,sb = mesh_of(75, reg)
print(f"[FLOOR] frame 75 twice (LoRA): verts {na:,} vs {nb:,} (d={abs(na-nb)})  "
      f"faces {fa:,} vs {fb:,} (d={abs(fa-fb)})  silIoU {iou(sa,sb):.5f}\n")
print(f'{"fr":>4s} | {"LORA verts":>11s} {"faces":>11s} | {"FROZEN verts":>12s} {"faces":>11s} | {"IoU(L,F)":>9s}')
print('-'*72)
rows=[]
for fi in A.frames:
    lv,lf,ls = mesh_of(fi, reg)
    zv,zf,zs = mesh_of(fi, None)
    rows.append((fi,lv,lf,zv,zf,iou(ls,zs))); 
    print(f'{fi:4d} | {lv:11,d} {lf:11,d} | {zv:12,d} {zf:11,d} | {iou(ls,zs):9.5f}')
L=np.array([x[1] for x in rows]); F=np.array([x[2] for x in rows])
print('-'*72)
print(f"LoRA verts spread {L.max()-L.min():,} ({100*(L.max()-L.min())/L.mean():.2f}%)  "
      f"faces spread {F.max()-F.min():,}   non-determinism floor {abs(na-nb)} verts")
print(f"-> meshes across frames are {'DIFFERENT' if (L.max()-L.min()) > 10*max(abs(na-nb),1) else 'the same within noise'}")
