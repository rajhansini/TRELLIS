"""
measure_crossing.py
-------------------
THE WITNESS.  Is the white produced by degenerate SDF zero-crossings?

THE CLAIM BEING TESTED
  A vertex colour is not a value the network emits. FlexiCubes finds where the
  SDF crosses zero along a cube edge and BLENDS the two corner colours there
  (flexicubes.py:223-234, _linear_interp):

      t      = s0 / (s0 - s1)              s0, s1 = the two corner SDFs
      colour = (1-t)*c0 + t*c1             then sigmoid  (flexicubes.py:94)

  On a clean mid-edge cut, t ~ 0.5 and the two corners AVERAGE -- independent
  per-corner noise partially cancels. When the surface grazes a corner, t -> 0
  or 1 and there is NO averaging: that one corner's value is expressed raw.

  So the prediction is:
      white-producing edges have SMALL corner-ness = min(t, 1-t)
      body edges have corner-ness near 0.5
  and the interpolated logit exceeds +4 preferentially where corner-ness is low.

  If corner-ness does NOT separate them, the mechanism is wrong and the white
  comes from somewhere else.

WHAT IS MEASURED, exactly
  out_layer output per cube, 101 channels (cube2mesh.py LAYOUTS):
      [0:8]    sdf     -- the 8 cube corners.  sdf_bias = -1/res is ADDED by
                          SparseFeatures2Mesh (cube2mesh.py:127) before use, so
                          it is added here too.
      [53:101] colour  -- 8 corners x 6; the first 3 of each 6 are RGB logits.

  Cube topology, verbatim from flexicubes.py:44:
      cube_edges = [0,1, 1,5, 4,5, 0,4, 2,3, 3,7, 6,7, 2,6, 2,0, 3,1, 7,5, 6,4]
  12 edges. For every edge whose two corner SDFs have OPPOSITE signs (i.e. the
  surface crosses it) we compute t, corner-ness, and the interpolated RGB logit.

  Frozen and adapted, same cubes, so the comparison is paired.

Usage:
  python .../measure_crossing.py --frame 1 --tau 4.0
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
ap.add_argument('--res', type=int, default=256, help='mesh extractor resolution')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'crossing')
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
from step8_decode_render.decode_render import normalize_slat

DEVICE = torch.device('cuda')
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED, STRUCT_SEED, NOISE_SEED = 'JeffreyXiang/TRELLIS-image-large', 42, 6
STEPS, RESCALE_T = 25, 3.0
_ts = np.linspace(1, 0, STEPS + 1); _ts = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SDF_E, DEF_E, W_E, C_E = 8, 32, 53, 101

# flexicubes.py:44 verbatim, reshaped to 12 (corner_a, corner_b) pairs
CUBE_EDGES = torch.tensor([0,1, 1,5, 4,5, 0,4, 2,3, 3,7, 6,7, 2,6,
                           2,0, 3,1, 7,5, 6,4], dtype=torch.long).reshape(12, 2)


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


def crossings(feats, res):
    """
    feats [Ncube, 101] straight from out_layer.

    Returns, for every cube edge whose two corner SDFs have opposite signs:
        t          where the zero sits on that edge
        cness      min(t, 1-t):  0.5 = clean mid-edge cut, 0 = grazing a corner
        logit      the interpolated RGB logit (max over R,G,B)
    """
    sdf = feats[:, :SDF_E] + (-1.0 / res)          # cube2mesh.py:127 adds sdf_bias
    col = feats[:, W_E:C_E].reshape(-1, 8, 6)[..., :3]     # [N, 8, 3] RGB logits
    a, b = CUBE_EDGES[:, 0].to(feats.device), CUBE_EDGES[:, 1].to(feats.device)
    s0, s1 = sdf[:, a], sdf[:, b]                          # [N, 12]
    cross = (s0 * s1) < 0                                  # sign change = surface here
    den = (s0 - s1)
    t = torch.where(cross & (den.abs() > 1e-12), s0 / den,
                    torch.full_like(s0, float('nan')))     # flexicubes.py:233
    cness = torch.minimum(t, 1.0 - t)
    c0, c1 = col[:, a, :], col[:, b, :]                    # [N, 12, 3]
    interp = (1 - t).unsqueeze(-1) * c0 + t.unsqueeze(-1) * c1
    logit = interp.amax(-1)                                # [N, 12]
    m = cross & torch.isfinite(t)
    return t[m].cpu().numpy(), cness[m].cpu().numpy(), logit[m].cpu().numpy()


def main():
    print('=' * 96)
    print(f'CROSSING WITNESS   frame {args.frame}   tau {args.tau}')
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
            S[name] = crossings(cap['f'].feats.detach().float(), args.res)
            print(f'[{name:8s}] {len(S[name][0]):,} surface crossings', flush=True)
            del slat; gc.collect(); torch.cuda.empty_cache()
    finally:
        h.remove()

    print('\n' + '=' * 96)
    print('CORNER-NESS  min(t, 1-t).  0.5 = clean mid-edge cut, 0 = grazes a corner')
    print(f"{'model':>9}{'n':>11}{'mean':>9}{'p01':>8}{'p10':>8}{'p50':>8}")
    for name in ('frozen', 'adapted'):
        _, cn, _ = S[name]
        print(f"{name:>9}{len(cn):>11,}{cn.mean():>9.4f}{np.percentile(cn,1):>8.4f}"
              f"{np.percentile(cn,10):>8.4f}{np.percentile(cn,50):>8.4f}")

    print('\n' + '-' * 96)
    print('THE TEST: do the SATURATED crossings have LOW corner-ness?')
    rows = []
    for name in ('frozen', 'adapted'):
        t, cn, lg = S[name]
        w = lg > args.tau
        r = dict(model=name, n=int(len(lg)), n_white=int(w.sum()),
                 frac=float(w.mean()),
                 cn_white=float(cn[w].mean()) if w.any() else float('nan'),
                 cn_rest=float(cn[~w].mean()),
                 cn_white_p50=float(np.median(cn[w])) if w.any() else float('nan'),
                 cn_rest_p50=float(np.median(cn[~w])))
        rows.append(r)
        print(f"\n  {name.upper()}: {r['n_white']:,} of {r['n']:,} crossings above "
              f"+{args.tau}  ({100*r['frac']:.4f}%)")
        if w.any():
            print(f"    corner-ness of SATURATED   mean {r['cn_white']:.4f}  "
                  f"median {r['cn_white_p50']:.4f}")
            print(f"    corner-ness of EVERY OTHER mean {r['cn_rest']:.4f}  "
                  f"median {r['cn_rest_p50']:.4f}")
            print(f"    ratio {r['cn_white']/max(r['cn_rest'],1e-9):.3f}x"
                  f"{'   <- saturated edges DO graze corners' if r['cn_white'] < 0.7*r['cn_rest'] else ''}")

    print('\n' + '-' * 96)
    print('SATURATION RATE vs corner-ness  (adapted)')
    t, cn, lg = S['adapted']
    print(f"{'corner-ness bin':>18}{'n':>12}{'% above tau':>14}{'mean logit':>12}")
    for lo, hi in [(0,.02),(.02,.05),(.05,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5)]:
        m = (cn >= lo) & (cn < hi)
        if m.sum() < 50: continue
        print(f"{f'[{lo:.2f},{hi:.2f})':>18}{int(m.sum()):>12,}"
              f"{100*(lg[m] > args.tau).mean():>13.4f}%{lg[m].mean():>12.4f}")

    json.dump({'tau': args.tau, 'rows': rows}, open(OUT / 'crossing.json', 'w'), indent=2)
    np.savez_compressed(OUT / 'crossing.npz',
                        **{f'{n}_{k}': v for n in S for k, v in
                           zip(('t','cness','logit'), S[n])})
    print('\n' + '=' * 96)
    print('READING IT')
    print('  If "% above tau" RISES as corner-ness falls, degenerate crossings are')
    print('  where the white is made, and the mechanism is confirmed.')
    print('  If it is FLAT across the bins, corner-ness has nothing to do with it.')
    print(f'\n  wrote {OUT}')
    print('=' * 96, flush=True)


if __name__ == '__main__':
    main()
