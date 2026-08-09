"""
measure_attn_route.py
---------------------
Does the per-voxel SCATTER come from attention filtering a shared edit?

WHAT IS ALREADY MEASURED
  The adapter's effect on out_layer colour logits is
      delta = delta_bar + eps,   delta_bar +2.31 (dark) .. +0.51 (bright),
                                 std(eps) ~ 1.6 IN EVERY BIN
  The scatter is 3x the systematic push at the bright end, and NOTHING predicts
  it: r = -0.31 against the frozen logit, < 0.14 against position, sdf, deform,
  flexicube weights, distance from centroid.

  White is logit > 4. Frozen never exceeds +2.57; adapted reaches +5.71 on
  0.162% of logits. Bright voxels sit 1.3 scatter-units from that line, dark
  ones 4.3. So the scatter is the mechanism -- but its ORIGIN is unexplained.

THE HYPOTHESIS THIS TESTS
  to_kv outputs k and v concatenated (Linear(1024, 2048)); the LoRA adds to both,
  so the value vectors change by

      dv = second half of ( B_kv A_kv c )        [L_img, 1024]

  dv is IDENTICAL for every voxel -- it is a function of the image tokens alone.
  But it arrives at each voxel through that voxel's own attention row:

      dh(voxel) = sum_j A[voxel, j] * dv_j

  Attention rows differ a lot (entropy 0.53-0.65, 42-80 distinct top-1 patches)
  and do NOT track geometry (d = 0.01 visible vs occluded). So one shared edit
  becomes 1,748 different results -- which is the right SHAPE for a scatter that
  no geometric predictor explains.

  UNVERIFIED SO FAR: whether the magnitude works out, and whether dh at a flow
  block survives the remaining blocks, 25 ODE steps and the frozen decoder to
  show up as delta at out_layer.

WHAT IS COMPUTED
  For every adapted block, at a chosen knot:
      dv           the value delta, from to_kv's LoRA, second half of the output
      A            the real attention rows, per head and head-mean
      dh_pred      A @ dv, per voxel        <- the prediction
  then, mapping each of the 467,264 cubes to its parent latent voxel
  (cube // 4, since 64^3 -> 256^3 is two 2x upsamplers, and 7301 * 64 = 467264),
  correlates ||dh_pred|| against the measured |delta| at out_layer.

  Also reports ||dv|| itself, so "the edit is tiny and cannot matter" is
  distinguishable from "the edit is large but uniformly distributed".

HOW TO READ IT
  strong correlation -> the scatter is attention filtering a shared edit; the
                        fix acts on to_kv.
  weak correlation   -> the scatter is manufactured inside the frozen decoder;
                        no adapter-side change reaches it.

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
ap.add_argument('--frame', type=int, default=75)
ap.add_argument('--tau', type=float, default=4.0)
ap.add_argument('--run', default='rung16e1_rembg_uniform_all_qkvo_r4_s6_0f8620e0')
ap.add_argument('--knot', type=int, default=12)
ap.add_argument('--blocks', type=int, nargs='+', default=[0, 5, 11, 17, 23])
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'attn_route')
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
    print('=' * 96)
    print(f'ATTENTION ROUTE:  does  A @ dv  predict the out_layer scatter?')
    print(f'  frame {args.frame}   run {args.run}')
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

    global LORA
    rd = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                    weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    print(f'[LORA] {args.run}  epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}\n')

    # ── 1. the measured target: |delta| at out_layer, per cube ──────────────
    cap = {}
    h = dec.out_layer.register_forward_hook(lambda m, i, o: cap.__setitem__('f', o))
    S = {}
    try:
        for name in ('frozen', 'adapted'):
            LORA = None if name == 'frozen' else reg
            slat = denoise(fm, noise, coords, cond)
            cap.clear()
            with torch.no_grad():
                dec(slat)
            S[name] = dict(feats=cap['f'].feats.detach().float().cpu(),
                           coords=cap['f'].coords.detach().cpu().numpy()[:, 1:])
            del slat; gc.collect(); torch.cuda.empty_cache()
    finally:
        h.remove()
    assert np.array_equal(S['frozen']['coords'], S['adapted']['coords'])
    rgb = lambda f: f[:, W_E:C_E].reshape(-1, 8, 6)[..., :3]
    dcube = (rgb(S['adapted']['feats']) - rgb(S['frozen']['feats'])).abs().amax(dim=(1,2)).numpy()
    cube_xyz = S['frozen']['coords']
    print(f'[TARGET] |delta| at out_layer for {len(dcube):,} cubes   '
          f'mean {dcube.mean():.3f}  p99 {np.percentile(dcube,99):.3f}')

    # ── 2. dv and the real attention rows, per adapted block ────────────────
    CAPA = {'want': None, 'A': None, 'coords': None}

    def _fwd_cap(module, x, context, idx):
        lb = LORA.get(idx) if LORA is not None else None
        q_sp = module._linear(module.to_q, x)
        if lb is not None and lb.get('to_q') is not None:
            q_sp = q_sp.replace(q_sp.feats + lb.get('to_q')(x.feats).to(q_sp.feats.dtype))
        q = module._reshape_chs(q_sp, (module.num_heads, -1))
        kv_t = module._linear(module.to_kv, context)
        if lb is not None and lb.get('to_kv') is not None:
            kv_t = kv_t + lb.get('to_kv')(context).to(kv_t.dtype)
        kv = module._fused_pre(kv_t, num_fused=2)
        if CAPA['want'] == idx:
            qf = q.feats.float(); k = kv[0, :, 0].float()
            A = torch.softmax(torch.einsum('nhd,lhd->nhl', qf, k) * (qf.shape[-1]**-0.5), -1)
            CAPA['A'] = A.mean(1).detach()          # [Ntok, L] head-mean
            CAPA['coords'] = x.coords.detach().cpu().clone()
            del qf, k, A
        hh = sparse_scaled_dot_product_attention(q, kv)
        hh = module._reshape_chs(hh, (-1,))
        out = module._linear(module.to_out, hh)
        if lb is not None and lb.get('to_out') is not None:
            out = out.replace(out.feats + lb.get('to_out')(hh.feats).to(out.feats.dtype))
        return out

    @contextmanager
    def cap_ctx(fmm):
        saved, i = {}, 0
        for blk in fmm.blocks:
            if not hasattr(blk, 'cross_attn'): continue
            ca = blk.cross_attn; saved[i] = ca.forward
            def _mk(m, idx):
                def _f(x, context=None): return _fwd_cap(m, x, context, idx)
                return _f
            ca.forward = _mk(ca, i); i += 1
        try: yield
        finally:
            j = 0
            for blk in fmm.blocks:
                if hasattr(blk, 'cross_attn') and j in saved:
                    blk.cross_attn.forward = saved[j]; j += 1

    LORA = reg
    knot = args.knot
    blocks = args.blocks
    print(f'\n[ROUTE] knot {knot} (t={T_PAIRS[knot][0]:.3f}), blocks {blocks}\n')
    print(f"{'blk':>4}{'||dv||':>10}{'||dv||/||v||':>14}{'||A@dv||':>11}"
          f"{'spread':>9}{'r vs |delta|':>14}")
    rows = []
    for b in blocks:
        # dv = second half of the to_kv LoRA output on the image tokens
        ca = [blk.cross_attn for blk in fm.blocks if hasattr(blk, 'cross_attn')][b]
        lb = reg.get(b)
        # the flow model runs fp16 (slat_flow_img_dit_L_64l8p2_fp16.json), so the
        # conditioning has to be cast to the weight dtype exactly as _linear does
        _wdt = ca.to_kv.weight.dtype
        with torch.no_grad():
            _c = cond.to(_wdt)
            v_frozen = ca.to_kv(_c)[0, :, 1024:]                      # [L, 1024]
            dkv = lb.get('to_kv')(_c)[0] if lb.get('to_kv') is not None else None
            dv = dkv[:, 1024:] if dkv is not None else torch.zeros_like(v_frozen)
        v_frozen = v_frozen.float(); dv = dv.float()
        # A at this block, this knot
        x = sp.SparseTensor(feats=noise.clone(), coords=coords)
        with torch.no_grad(), cap_ctx(fm):
            for j in range(STEPS):
                t, tp = T_PAIRS[j]
                CAPA['want'] = b if j == knot else None
                tt = torch.tensor([1000.0*t], device=DEVICE, dtype=torch.float32)
                vv = fm(x, tt, cond)
                CAPA['want'] = None
                x = x.replace(x.feats - (t - tp) * vv.feats)
        A = CAPA['A']; tokc = CAPA['coords'][:, 1:].numpy()
        assert A is not None, f'no attention captured at block {b}'
        dh = (A @ dv.float()).norm(dim=1).cpu().numpy()               # [Ntok]

        # tokens @ 32^3 -> cubes @ 256^3 is a factor 8 per axis
        fac = 256 // (int(tokc.max()) + 1)
        key = {tuple(c): i for i, c in enumerate(map(tuple, tokc))}
        idx = np.array([key.get(tuple(c // fac), -1) for c in cube_xyz])
        ok = idx >= 0
        r = float(np.corrcoef(dh[idx[ok]], dcube[ok])[0, 1])
        rows.append(dict(block=b, dv_norm=float(dv.norm()),
                         dv_rel=float(dv.norm()/v_frozen.norm()),
                         dh_mean=float(dh.mean()), dh_std=float(dh.std()), r=r,
                         mapped=int(ok.sum())))
        print(f"{b:>4}{dv.norm():>10.3f}{dv.norm()/v_frozen.norm():>13.3f}"
              f"{dh.mean():>11.3f}{dh.std():>9.3f}{r:>14.4f}", flush=True)
        del A, dh; gc.collect(); torch.cuda.empty_cache()

    json.dump(rows, open(OUT / 'attn_route.json', 'w'), indent=2)
    best = max(rows, key=lambda z: abs(z['r']))
    print('\n' + '=' * 96)
    print('VERDICT')
    print(f"  strongest block: {best['block']}   r = {best['r']:+.4f}")
    print(f"  ||dv||/||v||   : {best['dv_rel']:.4f}   (how big the value edit is,"
          f" relative to the values themselves)")
    print(f"  spread of A@dv : {best['dh_std']:.3f}  on mean {best['dh_mean']:.3f}")
    if abs(best['r']) > 0.35:
        print('  -> attention filtering a shared edit PREDICTS the scatter.')
        print('     the fix acts on to_kv.')
    else:
        print('  -> A @ dv does NOT predict the out_layer scatter (best |r| ='
              f" {abs(best['r']):.3f}).")
        print('     the scatter is manufactured downstream, inside the frozen')
        print('     decoder. no adapter-side change reaches it.')
    print('=' * 96, flush=True)


if __name__ == '__main__':
    main()
