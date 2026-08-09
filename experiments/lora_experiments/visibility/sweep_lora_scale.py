"""
sweep_lora_scale.py
-------------------
Apply the TRAINED adapter at strength s.  No retraining.

      delta(v) = s * B * A * x(v)        s=0 -> exactly frozen
                                         s=1 -> exactly rung16

This is the inference-time "LoRA weight" knob. DRaFT (Vicol et al., ICLR 2024,
arXiv 2309.17400) reports LoRA scaling as its most effective control against
reward over-optimisation, ahead of KL regularisation and early stopping. Our
LoRALayer has no alpha/r factor at all, so s=1 is what has been running.

WHAT IT DECIDES
  measured so far, frame 1, whose GT is a FLAT GREY teapot (saturation 0.046,
  0.00% orange) and which is in the TRAIN split:

      frozen (s=0)   mean albedo 0.121   white verts 0.00%
      rung16 (s=1)   mean albedo 0.312   white verts 2.87%
      GT             mean albedo 0.305

  The brightening 0.121 -> 0.312 is wanted; the 2.87% white is not. If they
  scale at DIFFERENT rates, some s in between gives the correction without the
  artefact and this is solved without training anything. If they scale TOGETHER,
  edit magnitude and artefact are inseparable and no strength setting helps --
  which is equally decisive and sends the fix somewhere else entirely.

  Implemented by scaling B, since delta = B @ (A x) is linear in B. A is left
  alone so the per-voxel mixing weights are unchanged and only the magnitude
  moves.

Original header follows.

Every comparison so far has been two histograms -- frozen's logits and adapted's
logits -- which contain no voxel identity and therefore cannot say what happened
to any individual voxel. Quoting

    frozen median -2.55 -> adapted median -1.52   +1.03
    frozen p99    +1.48 -> adapted p99    +3.35   +1.87
    frozen max    +2.57 -> adapted max    +5.71   +3.14

LOOKS like the top moved further than the middle, but those are PERCENTILES. The
voxel at frozen's p99 need not be the voxel at adapted's p99; the ranking may
have reshuffled entirely.

This computes the PAIRED difference, same voxel in both runs:

    delta(v) = logit_adapted(v) - logit_frozen(v)      for all 467,264 cubes

and plots it against the frozen logit.

    flat horizontal  -> a SHIFT. Every voxel moved the same amount, and the
                        white comes from somewhere other than the size of the edit.
    upward slope     -> a SCALE. Voxels that started high moved further, which
                        alone explains the tail crossing +4 while the median does
                        not, and the fix belongs on the multiplicative effect.

Also reports what else predicts delta -- frozen logit, feature norm, sdf, deform,
weights, position -- ALL of them, rather than one picked in advance.

Pairing is asserted, not assumed: both runs use the same fixed structure
(seed 42) and a deterministic decoder, so cube coords must match one-to-one.

Original header follows.

This asks no question and tests no hypothesis. Every previous measurement in
this project pooled logits into a histogram, which cannot say anything about
WHERE the white is. This one keeps the voxels separate and describes the set
that is actually offending.

WHAT IT DOES
  1. Captures out_layer's raw output for frame 75, frozen and adapted.
     Layout (cube2mesh.py): sdf 0:8 | deform 8:32 | weights 32:53 | color 53:101
     color is (8 corners, 6ch); the first 3 of each 6 are RGB logits.
     Vertex colour is sigmoid(logit)  --  flexicubes.py:94.

  2. Per cube, takes max |RGB logit| over its 8 corners.
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
ap.add_argument('--frame', type=int, default=1)
ap.add_argument('--tau', type=float, default=4.0)
ap.add_argument('--run', default='rung16e1_rembg_uniform_all_qkvo_r4_s6_0f8620e0')
ap.add_argument('--scales', type=float, nargs='+',
                default=[0.0, 0.25, 0.5, 0.75, 1.0],
                help='adapter strength s. 0 = frozen, 1 = as trained.')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'delta')
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




def main():
    print('=' * 92)
    print(f'LoRA STRENGTH SWEEP   frame {args.frame}   run {args.run}')
    print('=' * 92, flush=True)

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

    # GT for this frame, teapot pixels only
    _img = Image.open(GT_FRAMES_DIR / f'frame_{args.frame:04d}.png').convert('RGB') \
                .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt = torch.from_numpy(np.array(_img)).float().div(255.).permute(2,0,1).to(DEVICE)
    gt_tight = (gt.min(0).values < 0.95)
    gt_mean = float(gt[:, gt_tight].mean())
    print(f'[GT] frame {args.frame}: {int(gt_tight.sum()):,} teapot px   '
          f'mean {gt_mean:.4f}\n')

    global LORA
    rd = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                    weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    print(f'[LORA] epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}\n', flush=True)

    # keep pristine copies of B; delta = B @ (A x) is LINEAR in B, so scaling B
    # scales the edit exactly. A is untouched so the per-voxel mixing is fixed.
    B0 = {}
    for bi, blk in reg.blocks.items():
        for n in blk.targets:
            B0[(bi, n)] = getattr(blk, f'lora_{n}').B.detach().clone()

    def set_scale(s):
        with torch.no_grad():
            for (bi, n), b in B0.items():
                getattr(reg.blocks[bi], f'lora_{n}').B.copy_(s * b)

    hdr = (f"{'s':>6}{'||B||':>9}{'verts':>10}{'mean alb':>10}{'p99':>8}"
           f"{'WHITE%':>9}{'DARK%':>8}{'PSNR':>8}{'gt-err':>9}")
    print(hdr); print('-' * len(hdr))
    rows = []
    for s in args.scales:
        set_scale(s)
        LORA = reg if s != 0 else None
        slat = denoise(fm, noise, coords, cond)
        with torch.no_grad():
            mesh = dec(slat)[0]
        c = mesh.vertex_attrs[:, :3].detach().float()
        mx, mn = c.amax(1), c.amin(1)
        # render from the TRAINING camera and score against GT
        v_, f_ = mesh.vertices, mesh.faces
        e1, e2 = v_[f_[:,1]]-v_[f_[:,0]], v_[f_[:,2]]-v_[f_[:,0]]
        mesh.faces = f_[0.5*torch.cross(e1,e2,dim=1).norm(dim=1) > 1e-6]
        sv = mesh.vertices; mesh.vertices = align(mesh.vertices.detach())
        try:
            with torch.no_grad():
                r_ = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                                     return_types=['color','mask'])
        finally:
            mesh.vertices = sv
        mk = r_['mask'].unsqueeze(0)
        img = (r_['color']*mk + (1-mk)).clamp(0,1)
        sil = (r_['mask'] > 0.5)
        d = (img - gt)[:, sil]
        psnr = 10*math.log10(1.0/max(float(d.pow(2).mean()), 1e-12))
        bnorm = float(sum(float((s*b).norm()) for b in B0.values())/len(B0))
        rec = dict(s=s, b_norm=bnorm, verts=int(c.shape[0]),
                   mean_alb=float(c.mean()), p99=float(c.flatten().quantile(0.99)),
                   white=float((mn>0.9).float().mean()),
                   dark=float((mx<0.15).float().mean()), psnr=psnr,
                   gt_err=float(c.mean())-gt_mean)
        rows.append(rec)
        print(f"{s:>6.2f}{bnorm:>9.4f}{rec['verts']:>10,}{rec['mean_alb']:>10.4f}"
              f"{rec['p99']:>8.4f}{100*rec['white']:>8.2f}%{100*rec['dark']:>7.2f}%"
              f"{psnr:>8.3f}{rec['gt_err']:>+9.4f}", flush=True)
        del slat, mesh, c, r_
        gc.collect(); torch.cuda.empty_cache()

    set_scale(1.0)
    json.dump({'frame': args.frame, 'gt_mean': gt_mean, 'rows': rows},
              open(OUT / 'scale_sweep.json', 'w'), indent=2)

    print('\n' + '=' * 92)
    print('VERDICT')
    f_ = rows[0]; l_ = rows[-1]
    db = l_['mean_alb'] - f_['mean_alb']; dw = l_['white'] - f_['white']
    print(f"  brightening  {f_['mean_alb']:.4f} -> {l_['mean_alb']:.4f}   "
          f"(GT is {gt_mean:.4f})")
    print(f"  white        {100*f_['white']:.2f}% -> {100*l_['white']:.2f}%")
    # at which s does albedo first reach GT, and what is white there?
    hit = next((r for r in rows if r['mean_alb'] >= gt_mean), None)
    if hit:
        print(f"\n  albedo first reaches GT at s = {hit['s']:.2f}, "
              f"where white = {100*hit['white']:.2f}%  PSNR {hit['psnr']:.3f}")
    best = max(rows, key=lambda r: r['psnr'])
    print(f"  best PSNR    s = {best['s']:.2f}  ->  {best['psnr']:.3f}  "
          f"white {100*best['white']:.2f}%")
    # do they scale together?
    import numpy as _np
    a = _np.array([r['mean_alb'] for r in rows]); w = _np.array([r['white'] for r in rows])
    a_n = (a-a[0])/max(a[-1]-a[0],1e-12); w_n = (w-w[0])/max(w[-1]-w[0],1e-12)
    print(f"\n  normalised curves (0 at s=0, 1 at s=1):")
    for r, an, wn in zip(rows, a_n, w_n):
        print(f"    s={r['s']:.2f}   brightening {an:.3f}   white {wn:.3f}"
              f"   {'<- white LAGS, a middle s wins' if wn < an - 0.1 else ('<- white LEADS' if wn > an + 0.1 else '')}")
    print('=' * 92, flush=True)


if __name__ == '__main__':
    main()
