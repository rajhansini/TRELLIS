"""
measure_vertex_split.py
-----------------------
Is rung15's correction UNIFORM over the surface, or only right where the camera
looked?

TWO EXPLANATIONS THE RENDERS CANNOT SEPARATE
  Rendered brightness rises from 0.267 at the training view to 0.347 at 45-90
  degrees. Two stories fit that equally well:

   (a) the correction is NON-UNIFORM. The loss only ever saw the front, so only
       the front is constrained; the rest drifted bright.
   (b) the correction is UNIFORM, but rung15 correctly carries MORE CONTRAST
       than frozen (p5-p95 range 0.639 vs 0.414, against GT's 0.618). A
       higher-contrast texture shows more variation as the camera sweeps past
       different parts of the surface -- which is correct behaviour, not a bug.

  Both predict the same render. This distinguishes them PER VERTEX: label every
  vertex visible or occluded from the TRAINING camera by depth test, then compare
  albedo within each group.

    under (a): rung15's occluded vertices are systematically brighter than its
               visible ones, and frozen's are not.
    under (b): the visible/occluded gap is the same in both models, and rung15
               simply has a wider distribution.

  frozen is the control: it never saw the loss, so whatever visible/occluded gap
  it shows is the gap the geometry and the base model produce on their own.

Why there are no white patches on the rung15 renders

The rendered PNG cannot answer this. render_mesh does

    color * mask + (1 - mask)

so the background is white BY CONSTRUCTION, and the frame is 8-bit, so anything
above 1.0 has already been clipped to 255 by the time it reaches disk. From the
image alone, "a hole in the mesh" and "colour pushed past 1.0" look identical.

So this measures the VERTEX COLOURS THEMSELVES -- mesh.vertex_attrs[:, :3], the
albedo the renderer interpolates -- before any rasterisation, clamping or
quantisation. Frozen vs rung15, same frames, same seeds.

If rung15's albedo exceeds 1.0 where frozen's does not, the white patches are
saturation caused by the adapter, and the magnitude of the overshoot is
recoverable here and nowhere else.

This is angle-independent: it is a property of the mesh, not of the viewpoint,
so it is a separate axis of evidence from the orbit sweep.

Usage:
  python .../measure_color_overshoot.py --frames 1 30 60 75 90 120 150
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
ap.add_argument('--frames', type=int, nargs='+', default=[1, 30, 60, 75, 90, 120, 150])
ap.add_argument('--run', default='rung15v1_uniform_all_qkvo_r4_s6_2674ec70')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'color_overshoot')
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
from trellis.renderers.mesh_renderer import intrinsics_to_projection
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


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k = np.asarray(rv, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


_AS = float(ALIGN['scale'])
_AR = torch.tensor(rodrigues(ALIGN['rotvec']), dtype=torch.float32, device=DEVICE)
_AC = torch.tensor(ALIGN['centre'], dtype=torch.float32, device=DEVICE)
_AT = torch.tensor(ALIGN['translation'], dtype=torch.float32, device=DEVICE)


def align(v):
    """rung13's solved mesh->video registration. Same transform every rung
    renders through, so these labels match every other measurement."""
    return _AS * ((v - _AC) @ _AR.T) + _AC + _AT


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
    """modules.py:126-139 plus rung15's three deltas. Nothing else changed."""
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


def encode(dino, i):
    img = Image.open(GT_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


def denoise(fm, noise, coords, cond):
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad(), lora_ctx(fm):
        for t, tp in T_PAIRS:
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = fm(x, tt, cond)
            x = x.replace(x.feats - (t - tp) * v.feats)
    return normalize_slat(x)



def project(pts, flip_y):
    ext = EXTRINSICS.to(DEVICE).float(); intr = INTRINSICS.to(DEVICE).float()
    persp = intrinsics_to_projection(intr, 0.5, 3.0)
    homo = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    cam = homo @ ext.T
    clip = homo @ (persp @ ext).T
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * RENDER_RES
    yy = (ndc[:, 1] * 0.5 + 0.5)
    return px, ((1.0 - yy) if flip_y else yy) * RENDER_RES, cam[:, 2]


def sample_map(m, px, py):
    return m[py.round().long().clamp(0, RENDER_RES - 1),
             px.round().long().clamp(0, RENDER_RES - 1)]


def gate_proj(v, depth, mask):
    """nvdiffrast's row convention decided from the renderer's own depth buffer.
    Getting it wrong swaps visible and occluded and inverts the whole result."""
    res = {}
    for flip in (True, False):
        px, py, cz = project(v, flip)
        inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
        ok = inside & (sample_map(mask, px, py) > 0.5)
        if int(ok.sum()) < 100:
            res[flip] = 0.0; continue
        d = (cz[ok] - sample_map(depth, px, py)[ok]).abs()
        res[flip] = float((d < 1e-3).float().mean())
        print(f'  flip_y={str(flip):5s}  on-surface {res[flip]:.4f}')
    best = max(res, key=res.get); other = not best
    assert res[best] > 0.05, f'GATE-proj FAILED: best {res[best]:.4f}'
    assert res[best] > 3 * max(res[other], 1e-6), 'GATE-proj FAILED: not separable'
    print(f'[GATE-proj] PASSED -- flip_y={best}', flush=True)
    return best


def label_vertices(mesh, renderer, flip_holder):
    """Per-VERTEX visible/occluded from the training camera, by depth test."""
    v_, f_ = mesh.vertices, mesh.faces
    e1, e2 = v_[f_[:, 1]] - v_[f_[:, 0]], v_[f_[:, 2]] - v_[f_[:, 0]]
    mesh.faces = f_[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
    sv = mesh.vertices
    va = align(mesh.vertices.detach())
    mesh.vertices = va
    try:
        with torch.no_grad():
            r = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                                return_types=['depth', 'mask'])
    finally:
        mesh.vertices = sv
    depth, mask = r['depth'].squeeze(), r['mask'].squeeze()
    if flip_holder[0] is None:
        print('\n[GATE-proj] deciding row convention from the depth buffer')
        flip_holder[0] = gate_proj(va, depth, mask)
    px, py, cz = project(va, flip_holder[0])
    inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
    on_sil = inside & (sample_map(mask, px, py) > 0.5)
    # a vertex is VISIBLE iff it lies on the surface the camera actually sees
    vis = on_sil & ((cz - sample_map(depth, px, py)).abs() < 1e-3)
    del r, depth, mask
    return vis


def main():
    print('=' * 92)
    print('per-vertex albedo, VISIBLE vs OCCLUDED from the training camera')
    print(f'  frames {args.frames}   run {args.run}')
    print('=' * 92, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    assert coords.shape[0] == 7301
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass

    dino = pipe.models['image_cond_model'].to(DEVICE)
    toks = {f: encode(dino, f).cpu() for f in args.frames}
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
    assert max(float(getattr(b, f'lora_{n}').B.float().norm())
               for b in reg.blocks.values() for n in b.targets) > 1e-6
    print(f'\n[LORA] epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}', flush=True)

    flip = [None]; rows = []
    hdr = (f"{'frame':>5} {'model':>7} {'verts':>9} {'vis%':>7} | "
           f"{'VISIBLE mean':>13} {'OCCLUDED mean':>14} {'occ-vis':>9} {'ratio':>7} | "
           f"{'vis p99':>8} {'occ p99':>8}")
    print('\n' + hdr); print('-' * len(hdr))
    for f in args.frames:
        cond = toks[f].unsqueeze(0).to(DEVICE)
        for name in ('frozen', 'rung15'):
            LORA = None if name == 'frozen' else reg
            slat = denoise(fm, noise, coords, cond)
            with torch.no_grad():
                mesh = dec(slat)[0]
            vis = label_vertices(mesh, renderer, flip)
            c = mesh.vertex_attrs[:, :3].detach().float().mean(1)   # scalar albedo
            v_, o_ = c[vis], c[~vis]
            rec = dict(frame=f, model=name, n_vert=int(c.shape[0]),
                       frac_vis=float(vis.float().mean()),
                       vis_mean=float(v_.mean()), occ_mean=float(o_.mean()),
                       vis_p99=float(v_.quantile(0.99)), occ_p99=float(o_.quantile(0.99)),
                       vis_std=float(v_.std()), occ_std=float(o_.std()))
            rec['delta'] = rec['occ_mean'] - rec['vis_mean']
            rec['ratio'] = rec['occ_mean'] / max(rec['vis_mean'], 1e-8)
            rows.append(rec)
            print(f"{f:>5} {name:>7} {rec['n_vert']:>9,} {100*rec['frac_vis']:>6.1f}% | "
                  f"{rec['vis_mean']:>13.4f} {rec['occ_mean']:>14.4f} "
                  f"{rec['delta']:>+9.4f} {rec['ratio']:>7.3f} | "
                  f"{rec['vis_p99']:>8.4f} {rec['occ_p99']:>8.4f}", flush=True)
            del slat, mesh, c, vis
            gc.collect(); torch.cuda.empty_cache()

    json.dump(rows, open(OUT / 'vertex_split.json', 'w'), indent=2)
    fz = [r for r in rows if r['model'] == 'frozen']
    r15 = [r for r in rows if r['model'] == 'rung15']
    dz = np.mean([r['delta'] for r in fz]); d15 = np.mean([r['delta'] for r in r15])
    print('\n' + '=' * 92)
    print('VERDICT')
    print(f'  frozen  occluded - visible = {dz:+.4f}   <- the gap geometry alone produces')
    print(f'  rung15  occluded - visible = {d15:+.4f}')
    print(f'  excess attributable to the LoRA = {d15 - dz:+.4f}')
    print()
    if d15 - dz > 0.02:
        print('  (a) NON-UNIFORM: rung15 brightened the surface the camera never saw,')
        print('      beyond what frozen does. The correction did not propagate.')
    elif abs(d15 - dz) <= 0.02:
        print('  (b) UNIFORM: rung15 treats seen and unseen surface alike. The angular')
        print('      brightness variation is its higher contrast, not a coherence failure.')
    else:
        print('  rung15 made the UNSEEN surface DARKER than the seen one -- neither story.')
    print('=' * 92, flush=True)


if __name__ == '__main__':
    main()
