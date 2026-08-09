"""
measure_perhead_attention.py
----------------------------
PER-HEAD cross-attention analysis: which of the 16 heads carries usable
2D<->3D correspondence, in FROZEN TRELLIS and in rung15?

WHY THIS EXISTS
  Every attention number this project has produced so far was averaged over all
  16 heads (measure_xattn_tables.py: A.mean(dim=1)). Fuse3D (SIGGRAPH Asia 2025,
  arXiv 2602.17040) reports that only a minority of heads are usable --
  "we select the subset of heads 1, 5 and 13 that consistently produce focused
  and semantically meaningful activations. We then sum their logits to compute
  the final alignment score."

  If that holds here, the head-mean is 13 vague heads drowning 3 sharp ones, and
  our headline results -- Cohen's d = 0.01 between visible and occluded tokens,
  89-99% of attention mass on CLS -- are artefacts of the averaging rather than
  properties of the model. This script decides that.

  It also produces the thing the self-attention enhancement needs: a per-head
  ranking of which head best identifies WHICH VOXELS THE IMAGE VIEW REACHED.
  Without that there is nothing to target an enhancement at.

  Fuse3D's own head indices are NOT reused. They publish the indices but not the
  selection criterion, and head indexing is checkpoint-specific, so borrowing
  1/5/13 would be a guess dressed as a citation.

THE FOUR METRICS, per (frame, knot, block, head)

  1. prefix_mass / patch_mass
     Fraction of the row on [CLS]+4 [REG] versus on the 37x37=1369 spatial
     patches. A head at 99% prefix does no spatial work at all. Cheapest filter.

  2. entropy, top1_prob            "focused" in Fuse3D's phrasing
     Entropy over the patch distribution, /log(1369): 0 = one patch, 1 = uniform.
     top1_prob is the mass on the single strongest patch; flat = 1/1369 = 7.3e-4.

  3. coherence_ratio, spearman     "semantically meaningful"
     A head can be sharply peaked and still meaningless -- pointing confidently
     at unrelated patches. Coherence asks whether voxels NEAR EACH OTHER IN 3D
     attend to patches NEAR EACH OTHER IN THE IMAGE:
         coherence_ratio = mean 2D patch distance between 3D k-NN pairs
                         / mean 2D patch distance between random pairs
     < 1 means coherent; 1.0 means the head's choices carry no spatial structure.
     spearman is the rank correlation of 3D distance against 2D patch distance
     over sampled pairs -- same question, distribution-free.

  4. AUC_own_mass, AUC_dist        the metric the enhancement actually needs
     Per head, how well does a single scalar separate camera-VISIBLE tokens from
     OCCLUDED ones? Reported as AUC, not Cohen's d: AUC is scale-free and reads
     directly as "how good a voxel selector is this head". 0.5 = no information,
     1.0 = perfect. The best-AUC head IS the selector for a Fuse3D-style
     enhancement.

MEMORY
  Per-head A is [1748 tokens, 16 heads, 1374 image tokens] fp32 = 154 MB. All 24
  blocks at once would be 3.7 GB, so the statistics are computed INSIDE the
  capture hook and A is freed before the next block runs. One ODE pass therefore
  yields all 24 blocks x 16 heads.

  Capture is LOGGING ONLY: the block's real output still comes from the fused
  kernel, so the sampling trajectory is unperturbed.

GRIDS.  Three, and only the last is what cross-attention sees:
      467,264  decoder fine voxels
        7,301  sparse-structure voxels @ 64^3   (the flow model's INPUT)
        1,748  transformer tokens @ 32^3        (patch_size=2 downsamples first)
  All labels are on the 1,748 token grid, read from the live SparseTensor.

Usage:
  python .../measure_perhead_attention.py --frames 5 15 25 ... --knots 0 12 24
"""

import sys, os, argparse, json, math, gc, time
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
ap.add_argument('--frames', type=int, nargs='+',
                default=list(range(5, 151, 10)),   # the held-out split
                help='default = the 15 held-out frames [5,15,...,145]')
ap.add_argument('--knots',  type=int, nargs='+', default=[0, 12, 24])
ap.add_argument('--blocks', type=int, nargs='+', default=list(range(24)))
ap.add_argument('--run', default='rung15v1_uniform_all_qkvo_r4_s6_2674ec70')
ap.add_argument('--knn', type=int, default=8, help='3D neighbours for coherence')
ap.add_argument('--n-pairs', type=int, default=20000, help='random pairs for the baseline')
ap.add_argument('--tau-cells', type=float, default=1.5)
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'perhead_attention')
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
STEPS, RESCALE_T, FLOW_RES = 25, 3.0, 64
PATCH, GRID = 14, 518 // 14
N_PATCH = GRID * GRID
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


def align(v): return _AS * ((v - _AC) @ _AR.T) + _AC + _AT


def encode(dino, i):
    img = Image.open(GT_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB') \
               .resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


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
    """Decide nvdiffrast's row convention from the renderer's own depth buffer.
    A wrong choice here swaps front and back and inverts metric 4."""
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
    assert res[best] > 3 * max(res[other], 1e-6), \
        f'GATE-proj FAILED: not separable ({res[best]:.4f} vs {res[other]:.4f})'
    print(f'[GATE-proj] PASSED — flip_y={best}', flush=True)
    return best


# ── rung15's LoRA, rebuilt only far enough to load its checkpoint ────────────

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
            if i in set(active):
                mods[str(i)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            i += 1
        self.blocks = nn.ModuleDict(mods)
    def get(self, i):
        k = str(i); return self.blocks[k] if k in self.blocks else None


# ── per-head statistics, computed in-hook so A is never held across blocks ───

LORA = None
_CTX = {'want': set(), 'rows': [], 'coords': None, 'geo': None}


def _auc(score, pos):
    """
    AUC of `score` for separating pos from ~pos, via the rank (Mann-Whitney)
    identity. Scale-free, so it is comparable across heads whose scores live on
    different ranges — which Cohen's d is not.
    """
    npos = int(pos.sum()); nneg = int((~pos).sum())
    if npos == 0 or nneg == 0: return float('nan')
    r = torch.argsort(torch.argsort(score)).float() + 1.0
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _head_stats(A, geo):
    """
    A   [Ntok, H, L] softmaxed over L, per head (NOT averaged)
    geo dict with knn_idx, rand_i, rand_j, own_patch, visible
    returns list of per-head dicts
    """
    n_prefix = geo['n_prefix']
    own = geo['own_patch']; have = own >= 0; vis = geo['visible']
    out = []
    for h in range(A.shape[1]):
        a = A[:, h, :]
        pref = a[:, :n_prefix].sum(1)
        P = a[:, n_prefix:]
        Pn = P / P.sum(1, keepdim=True).clamp(min=1e-12)
        ent = -(Pn * (Pn + 1e-12).log()).sum(1) / math.log(P.shape[1])
        t1 = Pn.argmax(1); tp = Pn.max(1).values
        r, c = (t1 // GRID).float(), (t1 % GRID).float()

        # 3 — coherence: do 3D neighbours choose nearby patches?
        ki = geo['knn_idx']                                  # [N, k]
        dn = torch.sqrt((r[:, None] - r[ki]) ** 2 + (c[:, None] - c[ki]) ** 2).mean()
        i_, j_ = geo['rand_i'], geo['rand_j']
        dr = torch.sqrt((r[i_] - r[j_]) ** 2 + (c[i_] - c[j_]) ** 2).mean()
        ratio = float(dn / dr.clamp(min=1e-8))
        d2 = torch.sqrt((r[i_] - r[j_]) ** 2 + (c[i_] - c[j_]) ** 2)
        rk = lambda v: torch.argsort(torch.argsort(v)).float()
        a3, b3 = geo['rank_d3'], rk(d2)
        sp_ = float(((a3 - a3.mean()) * (b3 - b3.mean())).sum() /
                    (a3.std(unbiased=False) * b3.std(unbiased=False) * len(a3) + 1e-12))

        # 4 — does this head separate visible from occluded?
        op = own.clamp(min=0)
        ownm = torch.where(have, Pn.gather(1, op.unsqueeze(1)).squeeze(1),
                           torch.zeros_like(tp))
        dist = torch.sqrt((r - (op // GRID).float()) ** 2 + (c - (op % GRID).float()) ** 2)
        dist = torch.where(have, dist, torch.full_like(dist, float(GRID)))
        out.append({
            'prefix_mass': float(pref.mean()), 'patch_mass': float(1 - pref.mean()),
            'entropy': float(ent.mean()), 'top1_prob': float(tp.mean()),
            'coherence_ratio': ratio, 'spearman': sp_,
            'auc_own_mass': _auc(ownm, vis), 'auc_dist': _auc(-dist, vis),
            'n_distinct_top1': int(torch.unique(t1).numel()),
        })
    return out


def _cap_fwd(module, x, context, idx):
    lb = LORA.get(idx) if LORA is not None else None
    q_sp = module._linear(module.to_q, x)
    if lb is not None and lb.get('to_q') is not None:
        q_sp = q_sp.replace(q_sp.feats + lb.get('to_q')(x.feats).to(q_sp.feats.dtype))
    q = module._reshape_chs(q_sp, (module.num_heads, -1))
    kv_t = module._linear(module.to_kv, context)
    if lb is not None and lb.get('to_kv') is not None:
        kv_t = kv_t + lb.get('to_kv')(context).to(kv_t.dtype)
    kv = module._fused_pre(kv_t, num_fused=2)
    assert not module.qk_rms_norm, 'cross-attn qk_rms_norm on: A would be wrong'

    if idx in _CTX['want']:
        qf = q.feats.float(); k = kv[0, :, 0].float()
        A = torch.softmax(torch.einsum('nhd,lhd->nhl', qf, k) * (qf.shape[-1] ** -0.5),
                          dim=-1)                                   # [N, H, L]
        _CTX['coords'] = x.coords.detach().cpu().clone()
        if _CTX['geo'] is not None:
            for h, s in enumerate(_head_stats(A, _CTX['geo'])):
                _CTX['rows'].append({**s, 'block': idx, 'head': h})
        del qf, k, A          # freed before the next block allocates its own

    h = sparse_scaled_dot_product_attention(q, kv)
    h = module._reshape_chs(h, (-1,))
    out = module._linear(module.to_out, h)
    if lb is not None and lb.get('to_out') is not None:
        out = out.replace(out.feats + lb.get('to_out')(h.feats).to(out.feats.dtype))
    return out


@contextmanager
def cap_ctx(fm):
    saved, i = {}, 0
    for blk in fm.blocks:
        if not hasattr(blk, 'cross_attn'): continue
        ca = blk.cross_attn; saved[i] = ca.forward
        def _mk(m, idx):
            def _f(x, context=None): return _cap_fwd(m, x, context, idx)
            return _f
        ca.forward = _mk(ca, i); i += 1
    assert i == 24, f'expected 24 cross-attn blocks, found {i}'
    try: yield
    finally:
        j = 0
        for blk in fm.blocks:
            if hasattr(blk, 'cross_attn') and j in saved:
                blk.cross_attn.forward = saved[j]; j += 1


def run(fm, noise, coords, cond, knot, blocks):
    """One 25-step pass; captures every requested block at `knot`."""
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    _CTX['rows'] = []
    with torch.no_grad(), cap_ctx(fm):
        for j in range(STEPS):
            t, tp = T_PAIRS[j]
            _CTX['want'] = set(blocks) if j == knot else set()
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = fm(x, tt, cond)
            _CTX['want'] = set()
            x = x.replace(x.feats - (t - tp) * v.feats)
    return list(_CTX['rows']), x


def main():
    t0 = time.time()
    print('=' * 84)
    print('PER-HEAD cross-attention: which head carries usable 2D<->3D correspondence?')
    print(f'  frames {len(args.frames)}  knots {args.knots}  blocks {len(args.blocks)}')
    print(f'  run {args.run}')
    print('=' * 84, flush=True)

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
    L = toks[args.frames[0]].shape[0]; n_prefix = L - N_PATCH
    assert n_prefix in (1, 5), f'unexpected prefix count {n_prefix}'
    print(f'\n[DINO] {L} tokens = {n_prefix} prefix + {N_PATCH} patches')

    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)
    renderer = make_renderer()

    rd = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                    weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    assert max(float(getattr(b, f'lora_{n}').B.float().norm())
               for b in reg.blocks.values() for n in b.targets) > 1e-6, \
        'LoRA B all zero — adapter not loaded'
    print(f'[LORA] epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}', flush=True)

    global LORA
    rows, flip, TOKC, NTOK, INNER = [], None, None, None, None
    rng = np.random.default_rng(0)

    for fi, f in enumerate(args.frames, 1):
        cond = toks[f].unsqueeze(0).to(DEVICE)

        # frozen pass first: it defines the geometry and therefore the labels
        LORA = None
        _CTX['geo'] = None                      # first pass only learns the grid
        _, x_fz = run(fm, noise, coords, cond, args.knots[0], [args.blocks[0]])
        C = _CTX['coords']
        if TOKC is None:
            TOKC, NTOK = C, C.shape[0]
            INNER = FLOW_RES // fm.patch_size
            print(f'\n[TOKENS] {NTOK} tokens on a {INNER}^3 grid '
                  f'(patch_size={fm.patch_size}), not the 7301 voxels at {FLOW_RES}^3')
            assert int(TOKC[:, 1:].max()) < INNER
        else:
            assert torch.equal(C, TOKC), f'frame {f}: token coords changed'

        with torch.no_grad():
            mesh = dec(normalize_slat(x_fz))[0]
        v_, f_ = mesh.vertices, mesh.faces
        e1, e2 = v_[f_[:, 1]] - v_[f_[:, 0]], v_[f_[:, 2]] - v_[f_[:, 0]]
        mesh.faces = f_[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
        sv = mesh.vertices; mesh.vertices = align(mesh.vertices.detach())
        try:
            with torch.no_grad():
                r_ = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                                     return_types=['depth', 'mask'])
        finally:
            mesh.vertices = sv
        depth, maskm = r_['depth'].squeeze(), r_['mask'].squeeze()
        if flip is None:
            print('\n[GATE-proj] deciding nvdiffrast row convention from the depth buffer')
            flip = gate_proj(align(mesh.vertices.detach()), depth, maskm)

        cen3 = ((TOKC[:, 1:].float() + 0.5) / INNER - 0.5).to(DEVICE)
        px, py, cz = project(align(cen3), flip)
        inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
        on_sil = inside & (sample_map(maskm, px, py) > 0.5)
        tau = args.tau_cells * (_AS / INNER)
        visible = on_sil & ((cz - sample_map(depth, px, py)).abs() < tau)
        own = torch.full((NTOK,), -1, dtype=torch.long, device=DEVICE)
        own[inside] = ((py / PATCH).long().clamp(0, GRID - 1) * GRID +
                       (px / PATCH).long().clamp(0, GRID - 1))[inside]

        if fi == 1:
            vf = float(visible.float().mean())
            print(f'[VIS] visible {int(visible.sum())}/{NTOK} ({100*vf:.1f}%)')
            assert 0.02 < vf < 0.60, f'visible fraction {vf:.3f} implausible'
            # 3D geometry for coherence — fixed across frames, computed once
            D = torch.cdist(cen3, cen3)
            D.fill_diagonal_(float('inf'))
            knn_idx = D.topk(args.knn, largest=False).indices
            ii = torch.from_numpy(rng.integers(0, NTOK, args.n_pairs)).to(DEVICE)
            jj = torch.from_numpy(rng.integers(0, NTOK, args.n_pairs)).to(DEVICE)
            d3 = (cen3[ii] - cen3[jj]).norm(dim=1)
            GEO_FIX = dict(knn_idx=knn_idx, rand_i=ii, rand_j=jj, d3_rand=d3,
                           rank_d3=torch.argsort(torch.argsort(d3)).float(),
                           n_prefix=n_prefix)
            del D; gc.collect(); torch.cuda.empty_cache()

        geo = dict(GEO_FIX); geo.update(own_patch=own, visible=visible)
        del mesh, r_, depth, x_fz
        gc.collect(); torch.cuda.empty_cache()

        for k in args.knots:
            for name in ('frozen', 'rung15'):
                LORA = None if name == 'frozen' else reg
                _CTX['geo'] = geo
                rr, _x = run(fm, noise, coords, cond, k, args.blocks)
                for r2 in rr:
                    rows.append({**r2, 'model': name, 'frame': f, 'knot': k,
                                 't': float(T_PAIRS[k][0])})
                del _x, rr
                gc.collect(); torch.cuda.empty_cache()
        _CTX['geo'] = None
        print(f'  [{fi:2d}/{len(args.frames)}] frame {f:3d}  '
              f'{len(rows):,} rows  {(time.time()-t0)/60:.1f} min', flush=True)

    json.dump(rows, open(OUT / 'perhead_rows.json', 'w'))
    KEYS = ['prefix_mass', 'patch_mass', 'entropy', 'top1_prob', 'coherence_ratio',
            'spearman', 'auc_own_mass', 'auc_dist', 'n_distinct_top1']
    with open(OUT / 'perhead.csv', 'w') as fh:
        cols = ['model', 'frame', 'knot', 't', 'block', 'head'] + KEYS
        fh.write(','.join(cols) + '\n')
        for r2 in rows:
            fh.write(','.join(str(r2[c]) for c in cols) + '\n')

    # ── head ranking, averaged over frames/knots/blocks ─────────────────────
    import collections
    print('\n' + '=' * 84)
    for name in ('frozen', 'rung15'):
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r2 in rows:
            if r2['model'] != name: continue
            for kk in KEYS:
                agg[r2['head']][kk].append(r2[kk])
        print(f'\nHEAD RANKING — {name.upper()}   (mean over '
              f'{len(args.frames)} frames x {len(args.knots)} knots x '
              f'{len(args.blocks)} blocks)')
        print(f"{'head':>5}{'CLSmass':>9}{'entropy':>9}{'top1p':>9}"
              f"{'coher':>8}{'spear':>8}{'AUCown':>8}{'AUCdist':>9}{'distinct':>9}")
        order = sorted(agg, key=lambda h: -abs(np.nanmean(agg[h]['auc_own_mass']) - 0.5))
        for h in order:
            a = agg[h]
            print(f"{h:>5}{np.mean(a['prefix_mass']):>9.4f}{np.mean(a['entropy']):>9.4f}"
                  f"{np.mean(a['top1_prob']):>9.5f}{np.mean(a['coherence_ratio']):>8.3f}"
                  f"{np.mean(a['spearman']):>8.3f}{np.nanmean(a['auc_own_mass']):>8.3f}"
                  f"{np.nanmean(a['auc_dist']):>9.3f}{np.mean(a['n_distinct_top1']):>9.1f}")

    print('\n' + '=' * 84)
    print('READING IT')
    print('  CLSmass  1.0 = the head spends everything on CLS/REG, no spatial work.')
    print(f'  top1p    flat would be 1/{N_PATCH} = {1/N_PATCH:.2e}.')
    print('  coher    <1 means 3D neighbours pick nearby patches. 1.0 = no structure.')
    print('  spear    rank corr of 3D distance vs 2D patch distance. >0 = coherent.')
    print('  AUCown   0.5 = the head cannot tell visible from occluded.')
    print('           FURTHEST FROM 0.5 is the voxel selector the enhancement needs.')
    print(f'\n  rows sorted by |AUCown - 0.5|, most informative first')
    print(f'  wrote {OUT}/perhead.csv  ({len(rows):,} rows)  {(time.time()-t0)/60:.1f} min')
    print('=' * 84, flush=True)


if __name__ == '__main__':
    main()
