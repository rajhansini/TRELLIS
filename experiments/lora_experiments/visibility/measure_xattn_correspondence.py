"""
measure_xattn_correspondence.py
-------------------------------
Does the BACK of the teapot look at the video frame at all?

THE QUESTION, STATED PRECISELY

  Cross-attention is the only channel through which the frame reaches the 3D
  model.  In every one of the 24 flow blocks it forms

      A = softmax(q @ k.T / sqrt(d))        [Ntok tokens x N_tok image tokens]

  Row v is token v's distribution over the frame's DINOv2 tokens: 5 prefix
  (CLS + 4 registers) and 37x37 = 1369 spatial patches.

  THE BLOCKS DO NOT SEE THE 7301 SPARSE-STRUCTURE VOXELS.  patch_size=2 (see
  slat_flow_img_dit_L_64l8p2_fp16.json) makes input_blocks downsample 64^3 ->
  32^3 before block 0, so q has ~1748 rows on a 32^3 grid.  Cross-attention's
  spatial granularity is therefore 8x coarser IN VOLUME than the structure the
  pipeline sampled.  Measured, not assumed: an earlier version of this script
  labelled visibility at 64^3 and aborted on a shape mismatch (1748 vs 7301).
  The token coords are captured from the live SparseTensor and asserted to be
  identical across every capture.

  For a voxel the camera cannot see, that row is either
    - PEAKED   -> it is reading a specific place in the frame. An adapter here
                  has something to steer, and rung15's shortfall is a training
                  problem.
    - FLAT, or dumped on CLS -> it is reading no place at all. No adapter on
                  cross-attention can ever reach that voxel, and the fix is to
                  SUPPLY the correspondence (the Fuse3D move) rather than hope
                  the model has one.

  This script measures which.  It trains nothing and changes nothing.

WHY THE MATRIX HAS TO BE RECOMPUTED
  sparse_scaled_dot_product_attention (full_attn.py:30) goes straight from
  q,k,v to the output via xformers and never materialises A.  So A is formed
  here explicitly, FOR LOGGING ONLY -- the block's actual output still comes
  from the fused kernel, so the sampling trajectory is bit-identical to a normal
  run and nothing measured here is an artefact of the instrumentation.

VISIBILITY IS RECOMPUTED, NOT REUSED
  visibility/masks/visibility.npz labels 467,264 DECODER voxels via gradient
  flow.  Those are three grids away from where cross-attention lives:

      467,264 decoder fine voxels   (after both upsamplers)
        7,301 sparse-structure voxels at 64^3   (the flow model's INPUT)
        1,748 transformer tokens    at 32^3     (what cross-attention sees)

  So the existing labels cannot be joined to A at all.  Visibility is recomputed
  here on the TOKEN grid by depth test:

      token centre (world) = (idx + 0.5)/32 - 0.5        [inner resolution 32]
      -> rung13 alignment  -> camera space -> pixel
      visible  <=>  inside the silhouette AND |z_voxel - depth(pixel)| < tau

  tau = 1.5 cells.  "Visible" therefore means "this voxel's centre lies on the
  surface the camera actually sees", not "somewhere inside the object".

GATE-proj
  nvdiffrast's image-row convention (does row 0 mean NDC y=+1 or y=-1?) is the
  one place a silent sign error would invert every conclusion in this script.
  It is not assumed.  Both conventions are tried against the renderer's OWN
  depth buffer using the mesh's own vertices; the script requires exactly one to
  agree and prints which it picked.  If neither agrees it aborts.

Usage:
  python experiments/lora_experiments/visibility/measure_xattn_correspondence.py \
      --frames 1 75 150 --blocks 0 11 23 --knots 0 12 24
"""

import sys, os, argparse, json, math, gc
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent          # .../lora_experiments/visibility
_LEX  = _HERE.parent                             # .../lora_experiments
_ROOT = _LEX.parent.parent                       # .../TRELLIS
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, nargs='+', default=[1, 75, 150])
ap.add_argument('--blocks', type=int, nargs='+', default=[0, 11, 23],
                help='which of the 24 cross-attn blocks to read A from')
ap.add_argument('--knots',  type=int, nargs='+', default=[0, 12, 24],
                help='which denoising knots to read A at (0=t1.0, 24=t0.111)')
ap.add_argument('--run', default=None,
                help='optional run dir under runs/ whose lora_best.pt is loaded, '
                     'so the ADAPTED attention can be compared with the frozen '
                     'one. Omit for frozen only.')
ap.add_argument('--tau-cells', type=float, default=1.5,
                help='depth tolerance for "on the visible surface", in voxel cells')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'xattn_correspondence')
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

# ── constants, identical to the rungs so the voxel set matches exactly ────────
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
FLOW_RES    = 64           # SLatFlowModel resolution; asserted below
PATCH       = 14           # DINOv2 ViT-L/14
GRID        = 518 // PATCH  # 37
N_PATCH     = GRID * GRID   # 1369

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


# ── projection, verified against the renderer rather than assumed ────────────

def project(pts_world, flip_y):
    """
    world -> (pixel_x, pixel_y, camera_z), matching mesh_renderer.py:88-99.

    depth there is vertices_camera[..., 2], i.e. camera-space z, so camera_z
    returned here is directly comparable to the rendered depth buffer.
    """
    ext  = EXTRINSICS.to(DEVICE).float()
    intr = INTRINSICS.to(DEVICE).float()
    persp = intrinsics_to_projection(intr, 0.5, 3.0)          # near/far as in make_renderer
    homo  = torch.cat([pts_world, torch.ones_like(pts_world[:, :1])], dim=1)
    cam   = homo @ ext.T
    clip  = homo @ (persp @ ext).T
    w     = clip[:, 3:4].clamp(min=1e-8)
    ndc   = clip[:, :3] / w
    px    = (ndc[:, 0] * 0.5 + 0.5) * RENDER_RES
    yy    = (ndc[:, 1] * 0.5 + 0.5)
    py    = ((1.0 - yy) if flip_y else yy) * RENDER_RES
    return px, py, cam[:, 2]


def sample_map(m, px, py):
    """Nearest-neighbour sample of an [H,W] map at float pixel coords."""
    xi = px.round().long().clamp(0, RENDER_RES - 1)
    yi = py.round().long().clamp(0, RENDER_RES - 1)
    return m[yi, xi]


def gate_proj(mesh_v, depth, mask):
    """
    GATE-proj. Decide nvdiffrast's row convention from evidence.

    A mesh vertex that lies on the visible surface must project to a pixel whose
    rendered depth equals that vertex's own camera z. Try both conventions; the
    right one produces a large population of near-exact agreements, the wrong
    one does not. Require a clear winner.
    """
    res = {}
    for flip in (True, False):
        px, py, cz = project(mesh_v, flip)
        inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
        m = sample_map(mask, px, py) > 0.5
        d = sample_map(depth, px, py)
        ok = inside & m
        if ok.sum() < 100:
            res[flip] = (0.0, float('inf')); continue
        diff = (cz[ok] - d[ok]).abs()
        # fraction of on-silhouette vertices that sit exactly on the visible surface
        frac = float((diff < 1e-3).float().mean())
        res[flip] = (frac, float(diff.median()))
        print(f'  flip_y={str(flip):5s}  on-surface fraction={frac:.4f}  '
              f'median |z - depth|={float(diff.median()):.5f}')
    best = max(res, key=lambda f: res[f][0])
    other = not best
    assert res[best][0] > 0.05, (
        f'GATE-proj FAILED: neither row convention reproduces the renderer '
        f'(best on-surface fraction {res[best][0]:.4f}). The projection is wrong '
        f'and every visible/occluded label below would be meaningless.')
    assert res[best][0] > 3 * max(res[other][0], 1e-6), (
        f'GATE-proj FAILED: the two conventions are not separable '
        f'({res[best][0]:.4f} vs {res[other][0]:.4f}); cannot decide which is '
        f'right, so the labels cannot be trusted.')
    print(f'[GATE-proj] PASSED — flip_y={best}', flush=True)
    return best


# ── rung15's LoRA, reconstructed just enough to load its checkpoint ──────────
# Deliberately NOT `import rung15_v1_xattn_randt`: that module parses argv and
# rebuilds nvdiffrast at import time. These mirror its LoRALayer /
# XAttnLoRABundle / XAttnLoRARegistry attribute names exactly, which is what
# state_dict matching actually depends on.

class LoRALayer(torch.nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(r, i))
        self.B = torch.nn.Parameter(torch.zeros(o, r))
    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class XAttnLoRABundle(torch.nn.Module):
    def __init__(self, ca, rank, targets):
        super().__init__()
        self.targets = tuple(targets)
        for n in self.targets:
            lin = getattr(ca, n)
            setattr(self, f'lora_{n}', LoRALayer(lin.in_features, lin.out_features, rank))
    def get(self, n):
        return getattr(self, f'lora_{n}', None) if n in self.targets else None


class XAttnLoRARegistry(torch.nn.Module):
    def __init__(self, flow_model, active, rank, targets):
        super().__init__()
        mods, i = {}, 0
        for blk in flow_model.blocks:
            if not hasattr(blk, 'cross_attn'):
                continue
            if i in set(active):
                mods[str(i)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            i += 1
        self.blocks = torch.nn.ModuleDict(mods)
    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


# ── capture A without perturbing the trajectory ───────────────────────────────

_CAP = {'want': None, 'A': None, 'coords': None}
LORA = None   # set by --run


def _cap_fwd(module, x, context, idx):
    """
    Byte-for-byte modules.py:126-139 for the output, PLUS an explicit softmax
    for logging when this block is the requested one. The output still comes
    from the fused kernel, so instrumentation cannot alter the trajectory.
    """
    lb = LORA.get(idx) if LORA is not None else None

    q_sp = module._linear(module.to_q, x)
    if lb is not None and lb.get('to_q') is not None:
        q_sp = q_sp.replace(q_sp.feats + lb.get('to_q')(x.feats).to(q_sp.feats.dtype))
    q    = module._reshape_chs(q_sp, (module.num_heads, -1))
    kv_t = module._linear(module.to_kv, context)
    if lb is not None and lb.get('to_kv') is not None:
        kv_t = kv_t + lb.get('to_kv')(context).to(kv_t.dtype)
    kv   = module._fused_pre(kv_t, num_fused=2)
    assert not module.qk_rms_norm, 'cross-attn qk_rms_norm on: A would be wrong'

    if _CAP['want'] == idx:
        qf = q.feats.float()                       # [Ntok, H, hd]
        k  = kv[0, :, 0].float()                   # [L, H, hd]
        scale = qf.shape[-1] ** -0.5
        logits = torch.einsum('nhd,lhd->nhl', qf, k) * scale
        A = torch.softmax(logits, dim=-1)          # [Ntok, H, L]
        _CAP['A'] = A.mean(dim=1).detach().cpu()   # head-mean -> [Ntok, L]
        # The tokens the blocks see are NOT the 7301 input voxels. patch_size=2
        # downsamples 64^3 -> 32^3 in input_blocks before block 0, so q has
        # ~1748 rows on a 32^3 grid. Capturing the coords is the only way to
        # know where each attention row physically sits.
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


def run_to_knot(flow_model, noise, coords, cond, k, block=None):
    """Denoise 0..k-1, then evaluate at knot k with A captured from `block`."""
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad(), capture_ctx(flow_model):
        _CAP['want'] = None
        for j in range(k):
            t, tp = T_PAIRS[j]
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, tt, cond)
            x = x.replace(x.feats - (t - tp) * v.feats)
        _CAP['want'], _CAP['A'] = block, None
        t, _ = T_PAIRS[k]
        tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
        flow_model(x, tt, cond)
        _CAP['want'] = None
    A, C = _CAP['A'], _CAP['coords']
    _CAP['A'], _CAP['coords'] = None, None
    return A, C


# ── row statistics ────────────────────────────────────────────────────────────

def row_stats(A, n_prefix, own_patch):
    """
    A          [N, L] head-mean attention, rows sum to 1
    own_patch  [N] index into 0..1368 of the patch each voxel projects onto,
               or -1 if it projects outside the frame
    """
    pref = A[:, :n_prefix].sum(dim=1)                 # mass parked on CLS/registers
    P    = A[:, n_prefix:]                            # [N, 1369]
    tot  = P.sum(dim=1, keepdim=True).clamp(min=1e-12)
    Pn   = P / tot                                    # renormalised over patches
    ent  = -(Pn * (Pn + 1e-12).log()).sum(dim=1) / math.log(P.shape[1])
    top1 = Pn.argmax(dim=1)
    topp = Pn.max(dim=1).values

    r_t, c_t = top1 // GRID, top1 % GRID
    have = own_patch >= 0
    op = own_patch.clamp(min=0)
    r_o, c_o = op // GRID, op % GRID
    dist = torch.sqrt(((r_t - r_o).float() ** 2 + (c_t - c_o).float() ** 2))
    dist[~have] = float('nan')
    own_mass = torch.where(have, Pn.gather(1, op.unsqueeze(1)).squeeze(1),
                           torch.full_like(topp, float('nan')))
    return dict(prefix_mass=pref, entropy=ent, top1=top1, top1_prob=topp,
                dist_to_own=dist, own_mass=own_mass)


def summarize(st, vis):
    def g(v, m):
        x = v[m]
        x = x[~torch.isnan(x)]
        return (float(x.mean()), float(x.median())) if x.numel() else (float('nan'),) * 2
    out = {}
    for name, m in (('visible', vis), ('occluded', ~vis)):
        out[name] = {
            'n': int(m.sum()),
            'entropy_mean':      g(st['entropy'], m)[0],
            'top1_prob_mean':    g(st['top1_prob'], m)[0],
            'top1_prob_median':  g(st['top1_prob'], m)[1],
            'prefix_mass_mean':  g(st['prefix_mass'], m)[0],
            'dist_to_own_median': g(st['dist_to_own'], m)[1],
            'own_mass_mean':     g(st['own_mass'], m)[0],
        }
    return out


def main():
    print('=' * 74)
    print('cross-attention correspondence: does the unseen surface read the frame?')
    print(f'  frames {args.frames}   blocks {args.blocks}   knots {args.knots}')
    print(f'  output {OUT}')
    print('=' * 74, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipe.to(DEVICE)
    flow = pipe.models['slat_flow_model']
    dec  = pipe.models['slat_decoder_mesh']
    for p in flow.parameters(): p.requires_grad_(False)
    for p in dec.parameters():  p.requires_grad_(False)
    assert flow.resolution == FLOW_RES, f'flow res {flow.resolution} != {FLOW_RES}'

    ref = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs  = pipe.get_cond([ref])
    torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    N = coords.shape[0]
    assert N == 7301, f'N_vox={N} != 7301'
    del cs; gc.collect(); torch.cuda.empty_cache()
    print(f'\n[VOX] {N} flow voxels  coords range '
          f'{int(coords[:,1:].min())}..{int(coords[:,1:].max())} '
          f'(resolution {FLOW_RES})', flush=True)
    assert int(coords[:, 1:].min()) >= 0 and int(coords[:, 1:].max()) < FLOW_RES

    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    dino = pipe.models['image_cond_model'].to(DEVICE)
    toks = {f: encode_frame(dino, f).cpu() for f in args.frames}
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()
    L = toks[args.frames[0]].shape[0]
    n_prefix = L - N_PATCH
    print(f'[DINO] {L} tokens = {n_prefix} prefix (CLS/registers) + {N_PATCH} '
          f'patches ({GRID}x{GRID})', flush=True)
    assert n_prefix in (1, 5), f'unexpected prefix token count {n_prefix}'

    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(N, flow.in_channels, device=DEVICE)

    global LORA
    if args.run:
        rd = _LEX / 'runs' / args.run
        cfg = json.load(open(rd / 'config.json'))
        ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt',
                        map_location='cpu', weights_only=True)
        LORA = XAttnLoRARegistry(flow, cfg['active_blocks'], cfg['rank'],
                                 tuple(cfg['targets'])).to(DEVICE)
        missing, unexpected = LORA.load_state_dict(ck['registry_state'], strict=True), None
        LORA.eval()
        for p in LORA.parameters(): p.requires_grad_(False)
        bn = [float(m.B.float().norm()) for b in LORA.blocks.values()
              for _, m in [(n, getattr(b, f'lora_{n}')) for n in b.targets]]
        print(f'\n[LORA] {args.run}')
        print(f'  epoch {ck["epoch"]}  best_psnr {ck["best_psnr"]:.3f}  '
              f'blocks {len(cfg["active_blocks"])}  targets {tuple(cfg["targets"])}')
        print(f'  ||B|| mean={np.mean(bn):.4f} max={np.max(bn):.4f}  '
              f'(all-zero would mean the adapter is not loaded)', flush=True)
        assert np.max(bn) > 1e-6, 'LoRA loaded but B is all zero — no adapter effect'
    else:
        print('\n[LORA] none — measuring the FROZEN model', flush=True)

    # ── geometry: mesh, depth, silhouette, then the visible/occluded labels ──
    renderer = make_renderer()
    cond75 = toks[75].unsqueeze(0).to(DEVICE) if 75 in toks else \
             toks[args.frames[0]].unsqueeze(0).to(DEVICE)
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad():
        for t, tp in T_PAIRS:
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow(x, tt, cond75)
            x = x.replace(x.feats - (t - tp) * v.feats)
        mesh = dec(normalize_slat(x))[0]
    # Same preparation the rungs render through (rung13.render_mesh): drop
    # degenerate faces, then swap in the aligned vertices. Swapping on the real
    # mesh object rather than building a stand-in keeps every attribute the
    # renderer might touch, and keeps this silhouette comparable to the ones
    # every other measurement in this project was made against.
    _v, _f = mesh.vertices, mesh.faces
    e1 = _v[_f[:, 1]] - _v[_f[:, 0]]
    e2 = _v[_f[:, 2]] - _v[_f[:, 0]]
    mesh.faces = _f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
    mv = align(mesh.vertices.detach())
    _saved = mesh.vertices
    mesh.vertices = mv
    try:
        with torch.no_grad():
            r = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                                return_types=['depth', 'mask'])
    finally:
        mesh.vertices = _saved
    depth = r['depth'].squeeze()
    mask  = r['mask'].squeeze()
    print(f'\n[GEOM] mesh {mv.shape[0]:,} verts   silhouette {int((mask>0.5).sum()):,} px')

    print('\n[GATE-proj] deciding nvdiffrast row convention from the depth buffer')
    flip = gate_proj(mv, depth, mask)

    # Learn the token grid from the model rather than assuming it. patch_size=2
    # means the 24 blocks operate on a 32^3 grid of ~1748 tokens, not the 7301
    # input voxels -- so cross-attention's spatial granularity is 8x coarser in
    # volume than the sparse structure. Labelling visibility at 64^3 would
    # attach every attention row to the wrong place.
    probe_cond = toks[args.frames[0]].unsqueeze(0).to(DEVICE)
    _A0, TOKC = run_to_knot(flow, noise, coords, probe_cond,
                            args.knots[0], block=args.blocks[0])
    assert _A0 is not None and TOKC is not None, 'probe capture returned nothing'
    NTOK = _A0.shape[0]
    INNER = FLOW_RES // flow.patch_size
    print(f'\n[TOKENS] cross-attention sees {NTOK} tokens on a {INNER}^3 grid '
          f'(patch_size={flow.patch_size}), NOT the {N} input voxels at {FLOW_RES}^3')
    assert TOKC.shape[0] == NTOK, f'{TOKC.shape[0]} coords for {NTOK} rows'
    assert int(TOKC[:, 1:].min()) >= 0 and int(TOKC[:, 1:].max()) < INNER, (
        f'token coords range {int(TOKC[:,1:].min())}..{int(TOKC[:,1:].max())} '
        f'does not fit a {INNER}^3 grid')
    del _A0; gc.collect(); torch.cuda.empty_cache()

    centres = (TOKC[:, 1:].float() + 0.5) / INNER - 0.5
    centres = align(centres.to(DEVICE))
    px, py, cz = project(centres, flip)
    inside = (px >= 0) & (px < RENDER_RES) & (py >= 0) & (py < RENDER_RES)
    on_sil = inside & (sample_map(mask, px, py) > 0.5)
    dz     = cz - sample_map(depth, px, py)
    cell   = _AS / INNER
    tau    = args.tau_cells * cell
    visible = on_sil & (dz.abs() < tau)

    own_patch = torch.full((NTOK,), -1, dtype=torch.long, device=DEVICE)
    pr = (py / PATCH).long().clamp(0, GRID - 1)
    pc = (px / PATCH).long().clamp(0, GRID - 1)
    own_patch[inside] = (pr * GRID + pc)[inside]

    print(f'[VIS] cell={cell:.5f}  tau={tau:.5f} ({args.tau_cells} cells)')
    print(f'      on silhouette : {int(on_sil.sum()):5d} / {NTOK}  '
          f'({100*float(on_sil.float().mean()):.1f}%)')
    print(f'      VISIBLE       : {int(visible.sum()):5d} / {NTOK}  '
          f'({100*float(visible.float().mean()):.1f}%)')
    print(f'      occluded      : {int((~visible).sum()):5d} / {NTOK}  '
          f'({100*float((~visible).float().mean()):.1f}%)', flush=True)
    assert 0.02 < float(visible.float().mean()) < 0.60, (
        f'visible fraction {float(visible.float().mean()):.3f} is implausible; '
        f'the depth test is miscalibrated and the split cannot be trusted.')

    vis_cpu = visible.cpu()
    own_cpu = own_patch.cpu()

    # ── the sweep ────────────────────────────────────────────────────────────
    results = []
    for f in args.frames:
        cond = toks[f].unsqueeze(0).to(DEVICE)
        for k in args.knots:
            for b in args.blocks:
                A, C = run_to_knot(flow, noise, coords, cond, k, block=b)
                assert A is not None and A.shape == (NTOK, L), \
                    f'capture failed: got {None if A is None else tuple(A.shape)}'
                assert torch.equal(C, TOKC), (
                    'token coords changed between captures; the visible/occluded '
                    'labels would no longer line up with the attention rows')
                st = row_stats(A.float(), n_prefix, own_cpu)
                s  = summarize(st, vis_cpu)
                rec = {'frame': f, 'knot': k, 't': float(T_PAIRS[k][0]),
                       'block': b, **s}
                results.append(rec)
                v, o = s['visible'], s['occluded']
                print(f'  f{f:03d} k{k:02d}(t={T_PAIRS[k][0]:.3f}) blk{b:02d}  '
                      f'entropy vis={v["entropy_mean"]:.4f} occ={o["entropy_mean"]:.4f}  |  '
                      f'top1p vis={v["top1_prob_mean"]:.4f} occ={o["top1_prob_mean"]:.4f}  |  '
                      f'CLSmass vis={v["prefix_mass_mean"]:.3f} occ={o["prefix_mass_mean"]:.3f}',
                      flush=True)
                np.savez_compressed(
                    OUT / f'A_f{f:03d}_k{k:02d}_b{b:02d}.npz',
                    entropy=st['entropy'].numpy(), top1=st['top1'].numpy(),
                    top1_prob=st['top1_prob'].numpy(),
                    prefix_mass=st['prefix_mass'].numpy(),
                    dist_to_own=st['dist_to_own'].numpy(),
                    own_mass=st['own_mass'].numpy())
                del A, C, st
                gc.collect(); torch.cuda.empty_cache()

    json.dump({'n_vox': N, 'n_xattn_tokens': NTOK, 'inner_res': INNER,
               'patch_size': int(flow.patch_size), 'n_tokens': L, 'n_prefix': n_prefix, 'grid': GRID,
               'flip_y': bool(flip), 'tau_cells': args.tau_cells,
               'n_visible': int(visible.sum()), 'n_occluded': int((~visible).sum()),
               'frac_visible': float(visible.float().mean()),
               'results': results},
              open(OUT / 'correspondence.json', 'w'), indent=2)

    np.savez_compressed(OUT / 'voxel_labels.npz',
                        coords=TOKC.numpy(), visible=vis_cpu.numpy(),
                        own_patch=own_cpu.numpy(), dz=dz.cpu().numpy(),
                        px=px.cpu().numpy(), py=py.cpu().numpy())

    print('\n' + '=' * 74)
    print('HOW TO READ THIS')
    print('  entropy   0 = all attention on one patch, 1 = uniform over 1369.')
    print('            occluded ~= 1.0 means the unseen surface reads NO location.')
    print('  top1_prob mass on the single best patch. Near 1/1369 = 7.3e-04 is flat.')
    print('  CLSmass   attention parked on CLS/registers, i.e. on no location at all.')
    print(f'\n  wrote {OUT}/correspondence.json')
    print('=' * 74, flush=True)


if __name__ == '__main__':
    main()
