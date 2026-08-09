"""
measure_xattn_tables.py
-----------------------
How much does each 3D token attend to each image token — FROZEN vs rung15-v1,
over all 150 frames.  Produces the two tables for the paper.

WHY THIS IS A RE-RUN AND NOT A LOG READ
  Attention is not a stored quantity.  A = softmax(q @ k.T) is recomputed from
  (weights, input) on every forward pass, and TRELLIS's fused kernel
  (full_attn.py:30) never even materialises it.  Logging it during training was
  never an option: one A is 1748 x 1374 floats = 9.6 MB, and there are 24 blocks
  x 25 steps x 135 frames x 30 epochs of them.

  So the attention is regenerated from the three things that determine it, all
  of which were preserved:
      weights   lora_best.pt (epoch 29, held PSNR 21.953)
      input     structure seed 42 -> the same 7301 voxels; noise seed 6;
                the same DINOv2 tokens from the same 150 frames
      code      the same forward
  This is the identical computation the training run performed, not an estimate
  of it.  The only discrepancy is TRELLIS's own run-to-run non-determinism
  (xformers + spconv atomics), measured elsewhere at ~1e-2 on the SLaT.

THE GRIDS.  There are three, and only the last one is what cross-attention sees:
      467,264  decoder fine voxels          (what visibility.npz labels)
        7,301  sparse-structure voxels @ 64^3   (the flow model's INPUT)
        1,748  transformer tokens @ 32^3        (patch_size=2 downsamples first)
  Every label here is computed on the 1,748 token grid, read from the live
  SparseTensor and asserted identical across all captures and both models.

WHAT IS MEASURED, per token, per (frame, knot, block), per model
      prefix_mass   attention on CLS + 4 registers, i.e. on NO location
      patch_mass    1 - prefix_mass; what reaches the 37x37=1369 image patches
      entropy       over the patch distribution, /log(1369). 1.0 = uniform
      top1_prob     mass on the single strongest patch (flat = 1/1369 = 7.3e-4)
      dist_to_own   patch-units from the strongest patch to the patch this token
                    projects onto. Large = attends somewhere it is not.
      own_mass      attention on the token's own projected patch

DESIGN DECISIONS, stated because they shape the numbers
  1. ONE ODE pass per frame yields all 9 (knot, block) cells; the capture is
     read off as the pass sweeps by.  Capture is LOGGING ONLY -- each block's
     real output still comes from the fused kernel, so the trajectory is
     unperturbed and nothing here is an artefact of instrumentation.
  2. VISIBILITY IS PER FRAME.  These are 150 different meshes, so the visible
     token set genuinely changes frame to frame.
  3. The labels come from the FROZEN mesh and are applied to BOTH models.
     rung15 moves geometry (+1.27% verts), so using each model's own mesh would
     confound "attention changed" with "the labels changed".  The label
     disagreement induced by rung15's mesh is measured and reported separately
     (label_agreement in the json) so this choice can be audited.
  4. Raw A is written for frame 75 only, by default.  Per-token statistics for
     all 150 frames are 113 MB; the raw A matrices for all 150 would be 12.9 GB
     in fp16 -- large but NOT impossible, so --raw-frame -1 writes them all if
     the paper needs per-patch heatmaps beyond one frame.  (Logging A during
     TRAINING would have been ~23 TB, which is the actually-impossible one.)

Usage:
  python .../measure_xattn_tables.py --frames-all --blocks 0 11 23 --knots 0 12 24
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
ap.add_argument('--frames', type=int, nargs='+', default=None)
ap.add_argument('--frames-all', action='store_true', help='all 150')
ap.add_argument('--blocks', type=int, nargs='+', default=[0, 11, 23])
ap.add_argument('--knots',  type=int, nargs='+', default=[0, 12, 24])
ap.add_argument('--run', default='rung15v1_uniform_all_qkvo_r4_s6_2674ec70',
                help='the adapted model. Frozen is always measured alongside.')
ap.add_argument('--tau-cells', type=float, default=1.5)
ap.add_argument('--raw-frame', type=int, default=75,
                help='frame whose full A matrices are written (86 MB). '
                     '-1 writes every frame: 12.9 GB in fp16.')
ap.add_argument('--out', default=None)
args = ap.parse_args()

FRAMES = list(range(1, 151)) if args.frames_all else (args.frames or [1, 75, 150])
OUT = Path(args.out) if args.out else (_HERE / 'xattn_tables')
OUT.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, p):
        self._f = open(p, 'a', buffering=1)
    def write(self, m):
        sys.__stdout__.write(m); self._f.write(m)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(OUT / 'log.txt')
sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from contextlib import contextmanager

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.modules.sparse.attention import sparse_scaled_dot_product_attention
from trellis.renderers.mesh_renderer import intrinsics_to_projection
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
PRETRAINED  = 'JeffreyXiang/TRELLIS-image-large'
STRUCT_SEED = 42
NOISE_SEED  = 6
STEPS       = 25
RESCALE_T   = 3.0
FLOW_RES    = 64
PATCH       = 14
GRID        = 518 // PATCH        # 37
N_PATCH     = GRID * GRID         # 1369

_ts     = np.linspace(1, 0, STEPS + 1)
_ts     = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ALIGN = json.load(open(_HERE / 'alignment' / 'alignment.json'))


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k  = np.asarray(rv, dtype=np.float64) / th
    K  = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


_AS = float(ALIGN['scale'])
_AR = torch.tensor(rodrigues(ALIGN['rotvec']), dtype=torch.float32, device=DEVICE)
_AC = torch.tensor(ALIGN['centre'],      dtype=torch.float32, device=DEVICE)
_AT = torch.tensor(ALIGN['translation'], dtype=torch.float32, device=DEVICE)


def align(v):
    return _AS * ((v - _AC) @ _AR.T) + _AC + _AT


def encode_frame(dino, idx):
    img = Image.open(GT_FRAMES_DIR / f'frame_{idx:04d}.png').convert('RGB')
    img = img.resize((518, 518), Image.LANCZOS)
    a   = np.array(img).astype(np.float32) / 255.0
    x   = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


# ── projection, decided from the renderer's own output, never assumed ────────

def project(pts_world, flip_y):
    ext   = EXTRINSICS.to(DEVICE).float()
    intr  = INTRINSICS.to(DEVICE).float()
    persp = intrinsics_to_projection(intr, 0.5, 3.0)     # make_renderer's near/far
    homo  = torch.cat([pts_world, torch.ones_like(pts_world[:, :1])], dim=1)
    cam   = homo @ ext.T
    clip  = homo @ (persp @ ext).T
    ndc   = clip[:, :3] / clip[:, 3:4].clamp(min=1e-8)
    px    = (ndc[:, 0] * 0.5 + 0.5) * RENDER_RES
    yy    = (ndc[:, 1] * 0.5 + 0.5)
    py    = ((1.0 - yy) if flip_y else yy) * RENDER_RES
    return px, py, cam[:, 2]


def sample_map(m, px, py):
    xi = px.round().long().clamp(0, RENDER_RES - 1)
    yi = py.round().long().clamp(0, RENDER_RES - 1)
    return m[yi, xi]


def gate_proj(mesh_v, depth, mask):
    """
    nvdiffrast's image-row convention is the one silent sign error that would
    swap front and back and invert every conclusion. Decide it from evidence: a
    vertex on the visible surface must project to a pixel whose rendered depth
    equals its own camera z.
    """
    res = {}
    for flip in (True, False):
        px, py, cz = project(mesh_v, flip)
        inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
        ok = inside & (sample_map(mask, px, py) > 0.5)
        if int(ok.sum()) < 100:
            res[flip] = (0.0, float('inf')); continue
        diff = (cz[ok] - sample_map(depth, px, py)[ok]).abs()
        res[flip] = (float((diff < 1e-3).float().mean()), float(diff.median()))
        print(f'  flip_y={str(flip):5s}  on-surface={res[flip][0]:.4f}  '
              f'median|z-depth|={res[flip][1]:.5f}')
    best  = max(res, key=lambda f: res[f][0])
    other = not best
    assert res[best][0] > 0.05, (
        f'GATE-proj FAILED: neither convention reproduces the renderer '
        f'(best {res[best][0]:.4f}); every visible/occluded label would be junk.')
    assert res[best][0] > 3 * max(res[other][0], 1e-6), (
        f'GATE-proj FAILED: conventions not separable '
        f'({res[best][0]:.4f} vs {res[other][0]:.4f}).')
    print(f'[GATE-proj] PASSED — flip_y={best}', flush=True)
    return best


# ── rung15's LoRA, rebuilt only far enough to load its checkpoint ────────────
# NOT `import rung15_v1_xattn_randt`: that parses argv and rebuilds nvdiffrast
# at import. Attribute names mirror it exactly, which is all state_dict needs.

class LoRALayer(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, i))
        self.B = nn.Parameter(torch.zeros(o, r))
    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


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
    def __init__(self, flow_model, active, rank, targets):
        super().__init__()
        mods, i = {}, 0
        for blk in flow_model.blocks:
            if not hasattr(blk, 'cross_attn'):
                continue
            if i in set(active):
                mods[str(i)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            i += 1
        self.blocks = nn.ModuleDict(mods)
    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


# ── capture, without perturbing the trajectory ───────────────────────────────

_CAP = {'want': set(), 'store': {}, 'coords': None}
LORA = None


def _cap_fwd(module, x, context, idx):
    """
    modules.py:126-139 for the output, plus the three rung15 deltas when a LoRA
    is loaded, plus an explicit softmax when this block is being captured.
    The output always comes from the fused kernel, so capture cannot alter the
    trajectory.
    """
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

    if idx in _CAP['want']:
        qf = q.feats.float()               # [Ntok, H, hd]
        k  = kv[0, :, 0].float()           # [L, H, hd]
        logits = torch.einsum('nhd,lhd->nhl', qf, k) * (qf.shape[-1] ** -0.5)
        A = torch.softmax(logits, dim=-1)
        _CAP['store'][idx] = A.mean(dim=1).detach()      # head-mean, kept on GPU
        _CAP['coords'] = x.coords.detach().cpu().clone()
        del qf, k, logits, A

    h = sparse_scaled_dot_product_attention(q, kv)
    h = module._reshape_chs(h, (-1,))
    out = module._linear(module.to_out, h)
    if lb is not None and lb.get('to_out') is not None:
        out = out.replace(out.feats + lb.get('to_out')(h.feats).to(out.feats.dtype))
    return out


@contextmanager
def capture_ctx(flow_model):
    saved, i = {}, 0
    for blk in flow_model.blocks:
        if not hasattr(blk, 'cross_attn'):
            continue
        ca = blk.cross_attn
        saved[i] = ca.forward

        def _mk(mod, idx):
            def _f(x, context=None):
                return _cap_fwd(mod, x, context, idx)
            return _f

        ca.forward = _mk(ca, i)
        i += 1
    assert i == 24, f'expected 24 cross-attn blocks, found {i}'
    try:
        yield
    finally:
        j = 0
        for blk in flow_model.blocks:
            if hasattr(blk, 'cross_attn') and j in saved:
                blk.cross_attn.forward = saved[j]
                j += 1


def run_capture(flow, noise, coords, cond, knots, blocks):
    """
    ONE 25-step pass. Captures every (knot, block) cell as the pass sweeps by,
    and returns the final latent so the same pass also yields the mesh.
    """
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    cells, kset, bset = {}, set(knots), set(blocks)
    with torch.no_grad(), capture_ctx(flow):
        for j in range(STEPS):
            t, tp = T_PAIRS[j]
            _CAP['want']  = bset if j in kset else set()
            _CAP['store'] = {}
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow(x, tt, cond)
            for b, A in _CAP['store'].items():
                cells[(j, b)] = A
            _CAP['want'], _CAP['store'] = set(), {}
            x = x.replace(x.feats - (t - tp) * v.feats)
    return cells, x


# ── per-token statistics ─────────────────────────────────────────────────────

def row_stats(A, n_prefix, own_patch):
    """A [Ntok, L] rows sum to 1; own_patch [Ntok] in 0..1368 or -1."""
    pref = A[:, :n_prefix].sum(dim=1)
    P    = A[:, n_prefix:]
    Pn   = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ent  = -(Pn * (Pn + 1e-12).log()).sum(dim=1) / math.log(P.shape[1])
    top1 = Pn.argmax(dim=1)
    topp = Pn.max(dim=1).values
    have = own_patch >= 0
    op   = own_patch.clamp(min=0)
    dist = torch.sqrt(((top1 // GRID - op // GRID).float() ** 2 +
                       (top1 % GRID  - op % GRID ).float() ** 2))
    dist = torch.where(have, dist, torch.full_like(dist, float('nan')))
    own  = torch.where(have, Pn.gather(1, op.unsqueeze(1)).squeeze(1),
                       torch.full_like(topp, float('nan')))
    return {'prefix_mass': pref, 'patch_mass': 1.0 - pref, 'entropy': ent,
            'top1_prob': topp, 'top1': top1.float(),
            'dist_to_own': dist, 'own_mass': own}


STAT_KEYS = ['prefix_mass', 'patch_mass', 'entropy', 'top1_prob',
             'dist_to_own', 'own_mass']


def main():
    t00 = time.time()
    print('=' * 78)
    print('cross-attention tables: FROZEN vs rung15-v1, per token, per frame')
    print(f'  frames {len(FRAMES)}   blocks {args.blocks}   knots {args.knots}')
    print(f'  adapted run: {args.run}')
    print(f'  out {OUT}')
    print('=' * 78, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipe.to(DEVICE)
    flow = pipe.models['slat_flow_model']
    dec  = pipe.models['slat_decoder_mesh']
    for p in flow.parameters(): p.requires_grad_(False)
    for p in dec.parameters():  p.requires_grad_(False)
    assert flow.resolution == FLOW_RES

    ref = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs  = pipe.get_cond([ref])
    torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    NVOX = coords.shape[0]
    assert NVOX == 7301, f'N_vox={NVOX} != 7301'
    del cs; gc.collect(); torch.cuda.empty_cache()

    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    print(f'\n[DINO] encoding {len(FRAMES)} frames...', flush=True)
    dino = pipe.models['image_cond_model'].to(DEVICE)
    toks = {}
    for i, f in enumerate(FRAMES, 1):
        toks[f] = encode_frame(dino, f).cpu()
        if i % 50 == 0: print(f'  {i}/{len(FRAMES)}', flush=True)
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()
    L = toks[FRAMES[0]].shape[0]
    n_prefix = L - N_PATCH
    print(f'[DINO] {L} tokens = {n_prefix} prefix + {N_PATCH} patches ({GRID}x{GRID})')
    assert n_prefix in (1, 5), f'unexpected prefix count {n_prefix}'

    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(NVOX, flow.in_channels, device=DEVICE)
    renderer = make_renderer()

    # ── load the adapter ─────────────────────────────────────────────────────
    global LORA
    rd  = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck  = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                     weights_only=True)
    reg = XAttnLoRARegistry(flow, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True)
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)
    bn = [float(getattr(b, f'lora_{n}').B.float().norm())
          for b in reg.blocks.values() for n in b.targets]
    print(f'\n[LORA] {args.run}  epoch {ck["epoch"]}  '
          f'best_psnr {ck["best_psnr"]:.3f}')
    print(f'       blocks {len(cfg["active_blocks"])}  targets {tuple(cfg["targets"])}  '
          f'||B|| mean={np.mean(bn):.4f} max={np.max(bn):.4f}', flush=True)
    assert np.max(bn) > 1e-6, 'LoRA B is all zero — the adapter is not loaded'

    # ── the sweep ────────────────────────────────────────────────────────────
    CELLS = [(k, b) for k in args.knots for b in args.blocks]
    acc = {m: {c: {s: [] for s in STAT_KEYS} for c in CELLS}
           for m in ('frozen', 'rung15')}
    vis_frac, lbl_agree, TOKC, NTOK, INNER, flip = [], [], None, None, None, None
    per_frame_vis = {}

    for fi, f in enumerate(FRAMES, 1):
        cond = toks[f].unsqueeze(0).to(DEVICE)

        # ---- frozen pass: attention AND the mesh that defines the labels ----
        LORA = None
        cells_fz, x_fz = run_capture(flow, noise, coords, cond, args.knots, args.blocks)
        C = _CAP['coords']
        if TOKC is None:
            TOKC  = C
            NTOK  = TOKC.shape[0]
            INNER = FLOW_RES // flow.patch_size
            print(f'\n[TOKENS] cross-attention sees {NTOK} tokens on a {INNER}^3 '
                  f'grid (patch_size={flow.patch_size}), NOT the {NVOX} voxels '
                  f'at {FLOW_RES}^3', flush=True)
            assert int(TOKC[:, 1:].max()) < INNER
        else:
            assert torch.equal(C, TOKC), (
                f'frame {f}: token coords changed; labels would not line up')

        with torch.no_grad():
            mesh = dec(normalize_slat(x_fz))[0]
        _v, _f2 = mesh.vertices, mesh.faces
        e1 = _v[_f2[:, 1]] - _v[_f2[:, 0]]
        e2 = _v[_f2[:, 2]] - _v[_f2[:, 0]]
        mesh.faces = _f2[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
        saved = mesh.vertices
        mesh.vertices = align(mesh.vertices.detach())
        try:
            with torch.no_grad():
                r = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                                    return_types=['depth', 'mask'])
        finally:
            mesh.vertices = saved
        depth, maskm = r['depth'].squeeze(), r['mask'].squeeze()

        if flip is None:
            print('\n[GATE-proj] deciding nvdiffrast row convention from the depth buffer')
            flip = gate_proj(align(mesh.vertices.detach()), depth, maskm)

        centres = align(((TOKC[:, 1:].float() + 0.5) / INNER - 0.5).to(DEVICE))
        px, py, cz = project(centres, flip)
        inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
        on_sil = inside & (sample_map(maskm, px, py) > 0.5)
        tau    = args.tau_cells * (_AS / INNER)
        visible = on_sil & ((cz - sample_map(depth, px, py)).abs() < tau)
        own = torch.full((NTOK,), -1, dtype=torch.long, device=DEVICE)
        own[inside] = ((py / PATCH).long().clamp(0, GRID - 1) * GRID +
                       (px / PATCH).long().clamp(0, GRID - 1))[inside]
        vis_frac.append(float(visible.float().mean()))
        per_frame_vis[f] = visible.cpu().numpy()

        if fi == 1:
            assert 0.02 < vis_frac[0] < 0.60, (
                f'visible fraction {vis_frac[0]:.3f} implausible; depth test '
                f'miscalibrated and the whole split would be untrustworthy')

        del mesh, r, depth, x_fz
        gc.collect(); torch.cuda.empty_cache()

        # ---- adapted pass ---------------------------------------------------
        LORA = reg
        cells_r15, x_r15 = run_capture(flow, noise, coords, cond, args.knots, args.blocks)
        assert torch.equal(_CAP['coords'], TOKC), \
            f'frame {f}: rung15 token coords differ from frozen'

        # how much would rung15's OWN mesh have moved the labels? (audit only)
        if f == args.raw_frame:
            with torch.no_grad():
                m2 = dec(normalize_slat(x_r15))[0]
            _v2, _f3 = m2.vertices, m2.faces
            a1 = _v2[_f3[:, 1]] - _v2[_f3[:, 0]]
            a2 = _v2[_f3[:, 2]] - _v2[_f3[:, 0]]
            m2.faces = _f3[0.5 * torch.cross(a1, a2, dim=1).norm(dim=1) > 1e-6]
            sv = m2.vertices; m2.vertices = align(m2.vertices.detach())
            try:
                with torch.no_grad():
                    r2 = renderer.render(m2, EXTRINSICS.to(DEVICE),
                                         INTRINSICS.to(DEVICE),
                                         return_types=['depth', 'mask'])
            finally:
                m2.vertices = sv
            d2, k2 = r2['depth'].squeeze(), r2['mask'].squeeze()
            os2 = inside & (sample_map(k2, px, py) > 0.5)
            v2  = os2 & ((cz - sample_map(d2, px, py)).abs() < tau)
            lbl_agree.append(float((v2 == visible).float().mean()))
            print(f'\n[LABEL AUDIT] frame {f}: rung15\'s own mesh would agree with '
                  f'the frozen labels on {lbl_agree[-1]*100:.2f}% of tokens', flush=True)
            del m2, r2, d2, k2
        del x_r15
        gc.collect(); torch.cuda.empty_cache()

        # ---- statistics, both models, shared labels -------------------------
        for name, cells in (('frozen', cells_fz), ('rung15', cells_r15)):
            for c in CELLS:
                st = row_stats(cells[c].float(), n_prefix, own)
                for s in STAT_KEYS:
                    acc[name][c][s].append(st[s].cpu().numpy())
                if args.raw_frame == -1 or f == args.raw_frame:
                    np.savez_compressed(
                        OUT / f'A_raw_{name}_f{f:03d}_k{c[0]:02d}_b{c[1]:02d}.npz',
                        A=cells[c].cpu().numpy().astype(np.float16))
        del cells_fz, cells_r15
        gc.collect(); torch.cuda.empty_cache()

        if fi % 10 == 0 or fi == len(FRAMES):
            print(f'  [{fi:3d}/{len(FRAMES)}] frame {f:3d}  '
                  f'visible {vis_frac[-1]*100:5.1f}%  '
                  f'{(time.time()-t00)/60:.1f} min', flush=True)

    # ── aggregate ────────────────────────────────────────────────────────────
    VIS = np.stack([per_frame_vis[f] for f in FRAMES])          # [F, Ntok] bool
    rows = []
    for name in ('frozen', 'rung15'):
        for (k, b) in CELLS:
            for grp, m in (('visible', VIS), ('occluded', ~VIS)):
                rec = {'model': name, 'block': b, 'knot': k,
                       't': float(T_PAIRS[k][0]), 'group': grp,
                       'n_token_frames': int(m.sum())}
                for s in STAT_KEYS:
                    X = np.stack(acc[name][(k, b)][s])           # [F, Ntok]
                    sel = X[m]
                    sel = sel[~np.isnan(sel)]
                    rec[f'{s}_mean'] = float(sel.mean()) if sel.size else float('nan')
                    rec[f'{s}_std']  = float(sel.std())  if sel.size else float('nan')
                rows.append(rec)

    np.savez_compressed(
        OUT / 'per_token_stats.npz',
        frames=np.array(FRAMES), coords=TOKC.numpy(), visible=VIS,
        cells=np.array([[k, b] for k, b in CELLS]),
        **{f'{n}__{s}': np.stack([np.stack(acc[n][c][s]) for c in CELLS])
           for n in ('frozen', 'rung15') for s in STAT_KEYS})

    with open(OUT / 'tables.csv', 'w') as fh:
        cols = ['model', 'block', 'knot', 't', 'group', 'n_token_frames'] + \
               [f'{s}_{q}' for s in STAT_KEYS for q in ('mean', 'std')]
        fh.write(','.join(cols) + '\n')
        for r_ in rows:
            fh.write(','.join(str(r_[c]) for c in cols) + '\n')

    json.dump({'frames': FRAMES, 'n_vox': NVOX, 'n_xattn_tokens': NTOK,
               'inner_res': INNER, 'patch_size': int(flow.patch_size),
               'n_image_tokens': L, 'n_prefix': n_prefix, 'grid': GRID,
               'flip_y': bool(flip), 'tau_cells': args.tau_cells,
               'visible_frac_mean': float(np.mean(vis_frac)),
               'visible_frac_std': float(np.std(vis_frac)),
               'label_agreement_rung15_mesh': lbl_agree,
               'lora': {'run': args.run, 'epoch': int(ck['epoch']),
                        'best_psnr': float(ck['best_psnr'])},
               'rows': rows}, open(OUT / 'tables.json', 'w'), indent=2)

    # ── the two tables, printed ──────────────────────────────────────────────
    for name in ('frozen', 'rung15'):
        print('\n' + '=' * 78)
        print(f'TABLE — {name.upper()}   ({len(FRAMES)} frames, {NTOK} tokens each)')
        print('=' * 78)
        print(f"{'blk':>4} {'t':>6} {'group':>9} {'CLSmass':>9} {'patch':>7} "
              f"{'entropy':>8} {'top1p':>9} {'dist':>7} {'ownmass':>9}")
        for r_ in rows:
            if r_['model'] != name: continue
            print(f"{r_['block']:>4} {r_['t']:>6.3f} {r_['group']:>9} "
                  f"{r_['prefix_mass_mean']:>9.4f} {r_['patch_mass_mean']:>7.4f} "
                  f"{r_['entropy_mean']:>8.4f} {r_['top1_prob_mean']:>9.5f} "
                  f"{r_['dist_to_own_mean']:>7.2f} {r_['own_mass_mean']:>9.5f}")

    print('\n' + '=' * 78)
    print('READING IT')
    print('  CLSmass  attention on CLS/registers = on NO spatial location.')
    print(f'  top1p    flat would be 1/{N_PATCH} = {1/N_PATCH:.2e}.')
    print('  entropy  1.0 = uniform over all patches; 0 = one patch.')
    print('  dist     patch-units from the strongest patch to the token\'s own.')
    print(f'\n  visible tokens: {np.mean(vis_frac)*100:.1f}% +- '
          f'{np.std(vis_frac)*100:.1f}% across {len(FRAMES)} frames')
    print(f'  wrote {OUT}/tables.csv, tables.json, per_token_stats.npz')
    print(f'  total {(time.time()-t00)/60:.1f} min')
    print('=' * 78, flush=True)


if __name__ == '__main__':
    main()
