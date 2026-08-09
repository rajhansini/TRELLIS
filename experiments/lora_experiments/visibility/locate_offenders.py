"""
locate_offenders.py
-------------------
WHERE are the saturated voxels, and what are they made of?

This asks no question and tests no hypothesis. Every previous measurement in
this project pooled logits into a histogram, which cannot say anything about
WHERE the white is. This one keeps the voxels separate and describes the set
that is actually offending.

WHAT IT DOES
  1. Captures out_layer's raw output for frame 75, frozen and adapted.
     Layout (cube2mesh.py): sdf 0:8 | deform 8:32 | weights 32:53 | color 53:101
     color is (8 corners, 6ch); the first 3 of each 6 are RGB logits.
     Vertex colour is sigmoid(logit)  --  flexicubes.py:94.

  2. Per cube, takes max SIGNED RGB logit over its 8 corners.
     ONE-SIDED, and this matters: frozen's logits span -6.67 .. +2.57, so it
     legitimately uses deep negatives for black rock. An |logit| > 4 criterion
     flags 65.9% of FROZEN as offending -- more than the adapter -- because it
     counts blackness. White is logit > +4 and nothing else.
     Offender := that max exceeds TAU (default 4.0, where sigmoid goes flat:
     sigmoid(4)=0.982 with gradient 0.018; by 6 it is 0.0025).
     Measured previously: frozen never exceeds 2.94, adapted reaches 5.71.

  3. DESCRIBES the offending set without assuming anything about it:
       - how many, and what fraction
       - where in 3D: centroid, bounding box, spread per axis
       - CONNECTED COMPONENTS on the voxel grid (6-neighbourhood).
         One big blob and ten thousand specks are different phenomena and the
         histogram cannot tell them apart.
       - component size distribution
       - local density: how many of each offender's 26 neighbours are also
         offenders. Near 0 means isolated; near 26 means a solid region.
       - the offenders' own sdf and deform statistics vs everyone else's,
         since those come from the same 101 channels and are the geometry.

  4. RENDERS A HEATMAP MESH. The colour channels are overwritten with the
     excess (black = none, red = saturated) and pushed through the SAME
     to_representation, so the output is a picture of exactly which surface is
     offending, from 6 angles. This is the part that answers "where".

  5. Writes the offender coords to npz so anything else can be joined to them.

Usage:
  python .../locate_offenders.py --frame 75 --tau 4.0
"""

import sys, os, argparse, json, math, gc
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
ap.add_argument('--frame', type=int, default=75)
ap.add_argument('--tau', type=float, default=4.0)
ap.add_argument('--run', default='rung16e1_rembg_uniform_all_qkvo_r4_s6_0f8620e0')
ap.add_argument('--yaws', type=float, nargs='+', default=[0, 45, 90, 135, 180, 270])
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'offenders')
OUT.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, p): self._f = open(p, 'a', buffering=1)
    def write(self, m): sys.__stdout__.write(m); self._f.write(m)
    def flush(self): sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(OUT / 'log.txt'); sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from contextlib import contextmanager

sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.modules.sparse.attention import sparse_scaled_dot_product_attention
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED, STRUCT_SEED, NOISE_SEED = 'JeffreyXiang/TRELLIS-image-large', 42, 6
STEPS, RESCALE_T = 25, 3.0
_ts = np.linspace(1, 0, STEPS + 1); _ts = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ALIGN = json.load(open(_HERE / 'alignment' / 'alignment.json'))
SDF_E, DEF_E, W_E, C_E = 8, 32, 53, 101      # cube2mesh.py LAYOUTS


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k = np.asarray(rv, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


_AS = float(ALIGN['scale'])
_AR = torch.tensor(rodrigues(ALIGN['rotvec']), dtype=torch.float32, device=DEVICE)
_AC = torch.tensor(ALIGN['centre'], dtype=torch.float32, device=DEVICE)
_AT = torch.tensor(ALIGN['translation'], dtype=torch.float32, device=DEVICE)


def align(v): return _AS * ((v - _AC) @ _AR.T) + _AC + _AT


def orbit_extrinsics(yaw_deg, elev_deg=0.0, radius=2.0):
    """GATE-cam in render_rung14v1_video.py asserts yaw=0,elev=0 reproduces the
    confirmed EXTRINSICS, so this is the same camera family every render uses."""
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, fwd)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = right, -up, fwd
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32)


def encode(dino, i):
    img = Image.open(GT_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB') \
               .resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


class LoRALayer(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, i)); self.B = nn.Parameter(torch.zeros(o, r))
    def forward(self, x): return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class XAttnLoRABundle(nn.Module):
    def __init__(self, ca, rank, targets):
        super().__init__()
        self.targets = tuple(targets)
        for n in self.targets:
            lin = getattr(ca, n)
            setattr(self, f'lora_{n}', LoRALayer(lin.in_features, lin.out_features, rank))
    def get(self, n):
        return getattr(self, f'lora_{n}', None) if n in self.targets else None


class XAttnLoRARegistry(nn.Module):
    def __init__(self, fm, active, rank, targets):
        super().__init__()
        mods, i = {}, 0
        for blk in fm.blocks:
            if not hasattr(blk, 'cross_attn'): continue
            if i in set(active): mods[str(i)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            i += 1
        self.blocks = nn.ModuleDict(mods)
    def get(self, i):
        k = str(i); return self.blocks[k] if k in self.blocks else None


LORA = None


def _fwd(module, x, context, idx):
    lb = LORA.get(idx) if LORA is not None else None
    q_sp = module._linear(module.to_q, x)
    if lb is not None and lb.get('to_q') is not None:
        q_sp = q_sp.replace(q_sp.feats + lb.get('to_q')(x.feats).to(q_sp.feats.dtype))
    q = module._reshape_chs(q_sp, (module.num_heads, -1))
    kv_t = module._linear(module.to_kv, context)
    if lb is not None and lb.get('to_kv') is not None:
        kv_t = kv_t + lb.get('to_kv')(context).to(kv_t.dtype)
    kv = module._fused_pre(kv_t, num_fused=2)
    h = sparse_scaled_dot_product_attention(q, kv)
    h = module._reshape_chs(h, (-1,))
    out = module._linear(module.to_out, h)
    if lb is not None and lb.get('to_out') is not None:
        out = out.replace(out.feats + lb.get('to_out')(h.feats).to(out.feats.dtype))
    return out


@contextmanager
def lora_ctx(fm):
    saved, i = {}, 0
    for blk in fm.blocks:
        if not hasattr(blk, 'cross_attn'): continue
        ca = blk.cross_attn; saved[i] = ca.forward
        def _mk(m, idx):
            def _f(x, context=None): return _fwd(m, x, context, idx)
            return _f
        ca.forward = _mk(ca, i); i += 1
    assert i == 24
    try: yield
    finally:
        j = 0
        for blk in fm.blocks:
            if hasattr(blk, 'cross_attn') and j in saved:
                blk.cross_attn.forward = saved[j]; j += 1


def denoise(fm, noise, coords, cond):
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad(), lora_ctx(fm):
        for t, tp in T_PAIRS:
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = fm(x, tt, cond)
            x = x.replace(x.feats - (t - tp) * v.feats)
    return normalize_slat(x)


def components(coords_np):
    """
    Connected components of the offender set on the voxel grid, 6-neighbourhood.
    A single contiguous blob and ten thousand isolated specks produce the same
    histogram and mean very different things.
    """
    S = {tuple(c): i for i, c in enumerate(map(tuple, coords_np))}
    seen, comps = set(), []
    nbr = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for c in S:
        if c in seen: continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            p = stack.pop(); comp.append(p)
            for d in nbr:
                q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])
                if q in S and q not in seen:
                    seen.add(q); stack.append(q)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def neighbour_density(all_coords, off_mask):
    """For each offender, how many of its 26 neighbours are also offenders."""
    off = all_coords[off_mask]
    S = set(map(tuple, off))
    out = []
    for c in map(tuple, off):
        n = 0
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx==dy==dz==0: continue
                    if (c[0]+dx, c[1]+dy, c[2]+dz) in S: n += 1
        out.append(n)
    return np.array(out)


def main():
    print('=' * 96)
    print(f'LOCATING the saturated voxels.  frame {args.frame}   tau {args.tau}')
    print('=' * 96, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass

    dino = pipe.models['image_cond_model'].to(DEVICE)
    cond = encode(dino, args.frame).unsqueeze(0).to(DEVICE)
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)
    renderer = make_renderer()

    global LORA
    rd = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                    weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    print(f'[LORA] {args.run}  epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}\n')

    cap = {}
    h = dec.out_layer.register_forward_hook(lambda m, i, o: cap.__setitem__('f', o))

    store = {}
    try:
        for name in ('frozen', 'adapted'):
            LORA = None if name == 'frozen' else reg
            slat = denoise(fm, noise, coords, cond)
            cap.clear()
            with torch.no_grad():
                dec(slat)
            sptensor = cap['f']
            feats = sptensor.feats.detach().float()          # [Ncube, 101]
            cc = sptensor.coords.detach().cpu().numpy()[:, 1:]
            rgb = feats[:, W_E:C_E].reshape(-1, 8, 6)[..., :3]   # [Ncube, 8, 3]
            store[name] = dict(feats=feats, coords=cc,
                               maxlog=rgb.amax(dim=(1, 2)).cpu().numpy(),
                               sptensor=sptensor)
            print(f'[{name:8s}] {feats.shape[0]:,} cubes   '
                  f'max|logit| {store[name]["maxlog"].max():.3f}', flush=True)
            del slat
            gc.collect(); torch.cuda.empty_cache()
    finally:
        h.remove()

    # ── describe the offending set ───────────────────────────────────────────
    rep = {}
    for name in ('frozen', 'adapted'):
        d = store[name]
        m = d['maxlog'] > args.tau        # ONE-SIDED: white is logit > +tau
        n = int(m.sum())
        print('\n' + '-' * 96)
        print(f'{name.upper()}   offenders (max SIGNED RGB logit > +{args.tau}): '
              f'{n:,} of {len(m):,}  ({100*n/len(m):.4f}%)')
        r = dict(n=n, total=int(len(m)), frac=float(n / len(m)))
        if n == 0:
            print('  none.')
            rep[name] = r; continue

        oc = d['coords'][m]
        res = int(d['coords'].max()) + 1
        print(f'  voxel grid          : {res}^3')
        print(f'  centroid (grid)     : {oc.mean(0).round(1)}   all cubes: '
              f'{d["coords"].mean(0).round(1)}')
        print(f'  bbox                : {oc.min(0)} .. {oc.max(0)}   '
              f'span {oc.max(0)-oc.min(0)}')
        print(f'  spread (std/axis)   : {oc.std(0).round(1)}   all cubes: '
              f'{d["coords"].std(0).round(1)}')

        comps = components(oc)
        sizes = np.array([len(c) for c in comps])
        print(f'  CONNECTED COMPONENTS: {len(comps):,}')
        print(f'    largest           : {sizes[0]:,} voxels '
              f'({100*sizes[0]/n:.1f}% of all offenders)')
        print(f'    top 5 sizes       : {sizes[:5].tolist()}')
        print(f'    singletons        : {int((sizes==1).sum()):,} '
              f'({100*(sizes==1).sum()/len(sizes):.1f}% of components)')
        print(f'    median size       : {int(np.median(sizes))}')

        dens = neighbour_density(d['coords'], m)
        print(f'  neighbour density   : mean {dens.mean():.1f} of 26  '
              f'(0 = isolated specks, 26 = solid interior)')
        print(f'    fraction with 0   : {100*(dens==0).mean():.1f}%')
        print(f'    fraction with >=13: {100*(dens>=13).mean():.1f}%')

        sdf = d['feats'][:, :SDF_E].cpu().numpy()
        dfm = d['feats'][:, SDF_E:DEF_E].cpu().numpy()
        print(f'  their SDF  |mean|   : {np.abs(sdf[m]).mean():.4f}   '
              f'everyone else: {np.abs(sdf[~m]).mean():.4f}')
        print(f'  their DEFORM |mean| : {np.abs(dfm[m]).mean():.4f}   '
              f'everyone else: {np.abs(dfm[~m]).mean():.4f}')
        r.update(centroid=oc.mean(0).tolist(), bbox_min=oc.min(0).tolist(),
                 bbox_max=oc.max(0).tolist(), std=oc.std(0).tolist(),
                 n_components=len(comps), largest=int(sizes[0]),
                 singletons=int((sizes == 1).sum()),
                 nbr_density_mean=float(dens.mean()),
                 sdf_abs_off=float(np.abs(sdf[m]).mean()),
                 sdf_abs_rest=float(np.abs(sdf[~m]).mean()),
                 deform_abs_off=float(np.abs(dfm[m]).mean()),
                 deform_abs_rest=float(np.abs(dfm[~m]).mean()))
        np.savez_compressed(OUT / f'offenders_{name}.npz',
                            coords=oc, maxlog=d['maxlog'][m],
                            all_coords=d['coords'], all_maxlog=d['maxlog'])
        rep[name] = r

    # ── the picture: a mesh coloured by excess ───────────────────────────────
    print('\n' + '-' * 96)
    print('rendering the excess as a heatmap mesh (black = fine, red = saturated)')
    d = store['adapted']
    f2 = d['feats'].clone()
    excess = (d['feats'][:, W_E:C_E].reshape(-1, 8, 6)[..., :3]
              .amax(dim=2) - args.tau).clamp(min=0)                # [Ncube, 8] one-sided
    heat = (excess / max(float(excess.max()), 1e-6)).clamp(0, 1)   # 0..1 per corner
    col = f2[:, W_E:C_E].reshape(-1, 8, 6)
    col[..., 0] = 8.0 * heat - 4.0        # R logit: -4 (black) .. +4 (red)
    col[..., 1] = -4.0
    col[..., 2] = -4.0
    f2[:, W_E:C_E] = col.reshape(-1, 48)
    with torch.no_grad():
        mesh = dec.to_representation(d['sptensor'].replace(f2))[0]
    v_, fc = mesh.vertices, mesh.faces
    e1, e2 = v_[fc[:, 1]] - v_[fc[:, 0]], v_[fc[:, 2]] - v_[fc[:, 0]]
    mesh.faces = fc[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
    sv = mesh.vertices; mesh.vertices = align(mesh.vertices.detach())
    try:
        for yaw in args.yaws:
            with torch.no_grad():
                r_ = renderer.render(mesh, orbit_extrinsics(yaw).to(DEVICE),
                                     INTRINSICS.to(DEVICE),
                                     return_types=['color', 'mask'])
            mk = r_['mask'].unsqueeze(0)
            img = (r_['color'] * mk + (1 - mk)).clamp(0, 1)
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(OUT / f'heat_yaw{int(yaw):03d}.png')
            print(f'  yaw {yaw:5.1f}  -> heat_yaw{int(yaw):03d}.png', flush=True)
    finally:
        mesh.vertices = sv

    json.dump(rep, open(OUT / 'offenders.json', 'w'), indent=2)
    print(f'\nwrote {OUT}')
    print('  heat_yaw*.png   red = where the logits are saturated')
    print('  offenders_*.npz coords + logits, for joining to anything else')
    print('=' * 96, flush=True)


if __name__ == '__main__':
    main()
