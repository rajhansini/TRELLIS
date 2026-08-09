"""
measure_ghost.py
----------------
Quantify the artifact that block-level LoRA placements were rejected for.

WHY THIS EXISTS
  The placement ladder was decided on a qualitative look: block-level adapters
  (rung5.3/5.4/5.5) showed a faint shell hanging off the teapot, so the hook was
  moved to out_layer. No number was ever attached to that shell.

  Meanwhile, on the metric that WAS recorded, the block placements beat out_layer
  by ~11 dB (rung5.4 22.945, rung5.5-r16 21.418, v4 10.647, rung8 8.473) — same
  masked_psnr, same static mask, directly comparable.

  Worse, that PSNR is masked to the STATIC FROZEN SILHOUETTE. Anything hanging
  outside it contributes exactly zero. So the metric is blind to the artifact the
  block runs were rejected for, and the 11 dB gap is being read without knowing
  what it costs.

  This script attaches the missing number.

WHAT IT MEASURES, per arm, over 15 frames

  frac_outside_gt   |render ∧ ¬gt| / |render|
                    the fraction of rendered pixels sitting on GT background.
                    This is what was asked for. Note the frozen decoder already
                    scores ~18% here because TRELLIS's teapot differs from the
                    video's — so the number to read is DELTA vs frozen, not the
                    absolute value.

  frac_beyond_frozen  |render ∧ ¬dilate(frozen_mask, 2px)| / |render|
                    mass outside where the FROZEN mesh was, ignoring the
                    TRELLIS-vs-GT mismatch entirely. This isolates what the
                    adapter added. Frozen scores ~0 by construction.

  frac_detached     mass not in the largest connected component of render
                    a shell that separates from the body shows up here and
                    nowhere else.

  iou_gt            |render ∧ gt| / |render ∨ gt|
  sil_area, verts   block LoRAs feed out_layer, so they change sdf too —
                    they are NOT colour-only and their geometry moves.

ARMS
  frozen decoder, plus every adapter that has a checkpoint, block-level and
  out_layer alike, all decoded from the SAME SLaT cache with the SAME pinned
  frame-75 coords so the only difference is the adapter.

SAFETY
  Every state_dict loads with strict=True. If the module structure reconstructed
  here does not match the checkpoint exactly, it raises rather than silently
  applying a wrong adapter.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/measure_ghost.py
"""

import sys, os, gc, json, math, argparse
from contextlib import contextmanager
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_LORA = _HERE.parent
_ROOT = _LORA.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--stride',  type=int, default=10, help='probe every Nth frame')
ap.add_argument('--out-dir', default=None, type=Path)
ap.add_argument('--dilate',  type=int, default=2, help='px tolerance for frozen silhouette')
args = ap.parse_args()

GT_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
              '/outputs/teapot_lava_kling_premium'
              '/teapot_lava_kling_premium_front/all_frames_150')
SLAT_NPZ = _LORA / 'runs' / 'rung5_colonly_outlayer_r4_s6_c85c888f' / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'ghost_measurement')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES     = 150
GT_BG_THRESH = 0.95
DEC_DIM      = 768
COLOR_START, COLOR_END = 53, 101

# name -> (run dir, checkpoint kind)
ARMS = [
    ('rung5.3-late  blk 4-7  r4',  _LORA / 'runs' / 'rung5_53_mid_r4_s6_40243e3f',        'block'),
    ('rung5.3-late  blk 8-11 r4',  _LORA / 'runs' / 'rung5_53_late_r4_s6_1f298d4d',       'block'),
    ('rung5.4       blk 0-11 r4',  _LORA / 'runs' / 'rung5_54_r4_s6_31e5b3e8',            'block'),
    ('rung5.5       blk 8-11 r16', _LORA / 'runs' / 'rung5_55_r16_s6_9c47bd2e',           'block'),
    ('v4            out_layer',    _LORA / 'runs' / 'rung5_colonly_outlayer_r4_s6_c85c888f', 'outlayer'),
    ('rung8         out_layer',    _HERE / 'runs' / 'rung8_intersect_r4_s6_1b0d1d64',     'outlayer'),
    ('rung9         out_layer+geom', _HERE / 'runs' / 'rung9_geom-color_r4_s6_e44d8696',  'outlayer'),
]


def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p = torch.cuda.get_device_properties(0)
    arch_tag, arch_str = f'sm{p.major}{p.minor}', f'{p.major}.{p.minor}'
    local = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)
    try:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext(); del glctx
        print('[NVDIFF] OK', flush=True); return
    except Exception as e:
        print(f'[NVDIFF] FAILED: {e}', flush=True)
    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast broken for {arch_tag}')
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local, '--no-build-isolation',
             '--no-cache-dir', '--no-deps', '-q'],
            cwd=f'{src}/nvdiffrast', env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')


# ── LoRA modules — copied verbatim from rung5_subladder.py so strict=True
#    loading is a real structural check, not a formality ────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.scaling = 1.0
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # overwritten by the
        self.B = nn.Parameter(torch.zeros(out_dim, rank))  # strict load; kept so
                                                           # A is never raw memory
    def forward(self, x):
        return (((x.float() @ self.A.T) @ self.B.T) * self.scaling).to(x.dtype)


class DecBlockLoRABundle(nn.Module):
    def __init__(self, rank, dim=DEC_DIM):
        super().__init__()
        mlp_h = int(dim * 4)
        self.lora_qkv = LoRALayer(dim, 3 * dim, rank)
        self.lora_out = LoRALayer(dim, dim,     rank)
        self.lora_fc1 = LoRALayer(dim, mlp_h,   rank)
        self.lora_fc2 = LoRALayer(mlp_h, dim,   rank)


class DecLoRARegistry(nn.Module):
    def __init__(self, active_blocks, rank):
        super().__init__()
        self.active = set(active_blocks)
        self.blocks = nn.ModuleDict(
            {str(i): DecBlockLoRABundle(rank=rank) for i in active_blocks})

    def get(self, block_idx):
        # nn.ModuleDict has no .get() — this is rung5_subladder.py's form, verbatim
        key = str(block_idx)
        return self.blocks[key] if key in self.blocks else None


@contextmanager
def dec_block_lora_ctx(dec_model, registry):
    handles = []
    for i, block in enumerate(dec_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue

        def _qkv_hook(mod, inp, out, _lb=lb):
            return out + _lb.lora_qkv(inp[0]).to(out.dtype)

        def _out_hook(mod, inp, out, _lb=lb):
            return out + _lb.lora_out(inp[0]).to(out.dtype)

        def _fc1_hook(mod, inp, out, _lb=lb):
            return out.replace(out.feats + _lb.lora_fc1(inp[0].feats).to(out.feats.dtype))

        def _fc2_hook(mod, inp, out, _lb=lb):
            return out.replace(out.feats + _lb.lora_fc2(inp[0].feats).to(out.feats.dtype))

        handles.append(block.attn.to_qkv.register_forward_hook(_qkv_hook))
        handles.append(block.attn.to_out.register_forward_hook(_out_hook))
        handles.append(block.mlp.mlp[0].register_forward_hook(_fc1_hook))
        handles.append(block.mlp.mlp[2].register_forward_hook(_fc2_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


class OutLayerLoRA(nn.Module):
    """v4 / rung8 (out_dim 48) and rung9 (out_dim 80)."""
    def __init__(self, rank, in_dim, out_dim):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def _clamp_colour_only(rw, s, e, lo_v=-9.0, hi_v=8.0):
    lo, hi = max(s, COLOR_START), min(e, COLOR_END)
    if lo >= hi:
        return rw
    if lo == s and hi == e:
        return rw.clamp(lo_v, hi_v)
    out = rw.clone()
    out[:, lo - s:hi - s] = rw[:, lo - s:hi - s].clamp(lo_v, hi_v)
    return out


@contextmanager
def out_layer_lora_ctx(dec_model, lora, blocks, offs):
    def _hook(mod, inp, out):
        delta = lora(inp[0].feats)
        nf = out.feats.clone()
        for (s, e), off in zip(blocks, offs):
            w = e - s
            nf[:, s:e] = _clamp_colour_only(nf[:, s:e] + delta[:, off:off + w], s, e)
        return out.replace(nf)
    h = dec_model.out_layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


# ── rendering / metrics ───────────────────────────────────────────────────────

def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render(mesh, renderer):
    res = renderer.render(filter_degenerate_faces(mesh),
                          EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
    m = res['mask']
    c = (res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))).detach().clamp(0, 1)
    return c, (m > 0.5)


def gt_mask_of(fi):
    img = Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB') \
               .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    a = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1)
    return (a.min(dim=0).values < GT_BG_THRESH).to(DEVICE)


def frac_detached(rm_np):
    """Mass not in the largest connected component. A separated shell lands here."""
    lab, n = ndimage.label(rm_np)
    if n <= 1:
        return 0.0, n
    sizes = ndimage.sum(rm_np, lab, range(1, n + 1))
    return float(1.0 - sizes.max() / sizes.sum()), n


def measure(rm, gm, frozen_dil):
    rm_np = rm.cpu().numpy()
    tot = float(rm_np.sum())
    if tot < 1:
        return dict(frac_outside_gt=float('nan'), frac_beyond_frozen=float('nan'),
                    frac_detached=float('nan'), n_components=0, iou_gt=0.0, area=0)
    outside = float((rm & ~gm).sum().item()) / tot
    beyond  = float((rm_np & ~frozen_dil).sum()) / tot
    det, ncomp = frac_detached(rm_np)
    inter = float((rm & gm).sum().item()); union = float((rm | gm).sum().item())
    return dict(frac_outside_gt=outside, frac_beyond_frozen=beyond,
                frac_detached=det, n_components=int(ncomp),
                iou_gt=inter / max(union, 1.0), area=int(tot))


def build_arm(name, run_dir, kind, dec_model):
    """Returns (context_factory, n_params, note) or None if unavailable."""
    cfg_p, ck_p = run_dir / 'config.json', run_dir / 'lora_ckpts' / 'lora_best.pt'
    if not ck_p.exists():
        print(f'  [SKIP] {name}: no lora_best.pt at {ck_p}', flush=True)
        return None
    cfg = json.load(open(cfg_p))
    ck  = torch.load(ck_p, map_location='cpu', weights_only=True)

    if kind == 'block':
        reg = DecLoRARegistry(cfg['active_blocks'], cfg['rank']).to(DEVICE)
        reg.load_state_dict(ck['registry_state'], strict=True)   # <- structural check
        reg.eval()
        n = sum(p.numel() for p in reg.parameters())
        return (lambda: dec_block_lora_ctx(dec_model, reg), n,
                f"blocks {cfg['active_blocks']} r{cfg['rank']} ep{ck['epoch']}")

    blocks = [tuple(b) for b in ck['blocks']] if 'blocks' in ck else [(53, 101)]
    offs, o = [], 0
    for s, e in blocks:
        offs.append(o); o += e - s
    Bw = ck['lora_state']['B']
    assert Bw.shape[0] == o, f'{name}: B has {Bw.shape[0]} rows, blocks need {o}'
    lora = OutLayerLoRA(Bw.shape[1], 96, o).to(DEVICE)
    lora.load_state_dict(ck['lora_state'], strict=True)          # <- structural check
    lora.eval()
    n = sum(p.numel() for p in lora.parameters())
    return (lambda: out_layer_lora_ctx(dec_model, lora, blocks, offs), n,
            f"{blocks} r{Bw.shape[1]} ep{ck['epoch']}")


def main():
    frames = list(range(1, N_FRAMES + 1, args.stride))
    print('=' * 92)
    print('GHOST MEASUREMENT — how much rendered mass sits where it should not')
    print(f'  frames  : {frames}')
    print(f'  dilation: {args.dilate} px tolerance on the frozen silhouette')
    print('=' * 92, flush=True)

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] {slats.shape}  pinned coords {tuple(coords.shape)}', flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec  = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    for n in list(pipe.models.keys()):
        if n != 'slat_decoder_mesh':
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()
    renderer = make_renderer(DEVICE)

    def slat(fi):
        return sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))

    # ---- frozen reference first: its silhouette defines "beyond frozen" ----
    print('\n[FROZEN] reference pass', flush=True)
    frozen_masks, frozen_dil, gts, rows = {}, {}, {}, []
    fr = []
    for fi in frames:
        gm = gt_mask_of(fi); gts[fi] = gm
        with torch.no_grad():
            m = dec(slat(fi))[0]
        nv = int(m.vertices.shape[0])
        _, rm = render(m, renderer)
        frozen_masks[fi] = rm.clone()
        frozen_dil[fi] = ndimage.binary_dilation(
            rm.cpu().numpy(), iterations=args.dilate)
        st = measure(rm, gm, frozen_dil[fi]); st['frame'] = fi; st['verts'] = nv
        fr.append(st)
        del m
        gc.collect(); torch.cuda.empty_cache()

    def agg(lst, k):
        v = [x[k] for x in lst if not (isinstance(x[k], float) and math.isnan(x[k]))]
        return float(np.mean(v)) if v else float('nan')

    rows.append(dict(name='frozen decoder', params=0, note='no adapter',
                     **{k: agg(fr, k) for k in
                        ('frac_outside_gt', 'frac_beyond_frozen', 'frac_detached',
                         'n_components', 'iou_gt', 'area', 'verts')}))

    # ---- each adapter ----
    for name, run_dir, kind in ARMS:
        built = build_arm(name, run_dir, kind, dec)
        if built is None:
            continue
        ctx_factory, nparams, note = built
        print(f'\n[ARM] {name}   {note}   params={nparams:,}', flush=True)
        per = []
        for fi in frames:
            with torch.no_grad(), ctx_factory():
                m = dec(slat(fi))[0]
                nv = int(m.vertices.shape[0])
                _, rm = render(m, renderer)
            st = measure(rm, gts[fi], frozen_dil[fi]); st['frame'] = fi; st['verts'] = nv
            per.append(st)
            del m, rm
            gc.collect(); torch.cuda.empty_cache()
        rows.append(dict(name=name, params=nparams, note=note,
                         **{k: agg(per, k) for k in
                            ('frac_outside_gt', 'frac_beyond_frozen', 'frac_detached',
                             'n_components', 'iou_gt', 'area', 'verts')}))
        print(f'   outside_gt={rows[-1]["frac_outside_gt"]*100:5.2f}%  '
              f'beyond_frozen={rows[-1]["frac_beyond_frozen"]*100:5.2f}%  '
              f'detached={rows[-1]["frac_detached"]*100:5.2f}%  '
              f'IoU={rows[-1]["iou_gt"]:.4f}  verts={rows[-1]["verts"]:,.0f}', flush=True)

    # ---- report ----
    print('\n' + '=' * 108)
    print(f'{"arm":34s}{"params":>10s}{"outsideGT":>11s}{"beyondFrz":>11s}'
          f'{"detached":>10s}{"#cc":>5s}{"IoU":>8s}{"area":>9s}{"verts":>10s}')
    print('-' * 108)
    base = rows[0]['frac_outside_gt']
    for r in rows:
        print(f'{r["name"]:34s}{r["params"]:>10,}'
              f'{r["frac_outside_gt"]*100:>10.2f}%{r["frac_beyond_frozen"]*100:>10.2f}%'
              f'{r["frac_detached"]*100:>9.2f}%{r["n_components"]:>5.1f}'
              f'{r["iou_gt"]:>8.4f}{r["area"]:>9,.0f}{r["verts"]:>10,.0f}')
    print('-' * 108)
    print(f'frozen already sits {base*100:.2f}% outside GT (TRELLIS teapot != video '
          f'teapot). Read outsideGT as a DELTA vs that.')
    print('beyondFrz is the honest "what did the adapter add" column: frozen is 0 '
          'by construction.')
    print('=' * 108, flush=True)

    json.dump({'frames': frames, 'dilate_px': args.dilate, 'rows': rows},
              open(OUT_DIR / 'ghost_measurement.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "ghost_measurement.json"}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
