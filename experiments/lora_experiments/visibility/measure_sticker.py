"""
measure_sticker.py
------------------
Why does rung5_colonly's texture behave like a sticker? Measured on the vertices,
not inferred from renders.

WHAT IS ALREADY ESTABLISHED (not re-tested here)
  - There is no UV mapping. trellis/renderers/mesh_renderer.py only ever calls
    dr.interpolate(mesh.vertex_attrs[...]) — per-vertex colour interpolated across
    triangles. No texcoords, no texture image, no sampler. A UV bug is impossible.
  - Geometry is identical to frozen: 0/90 vertex mismatches and 0 px silhouette
    difference across a full orbit.

  So the ONLY thing that differs between the frozen render and the colonly render
  is a per-vertex RGB delta. This script measures that delta directly.

THE COMPETING EXPLANATIONS, AND WHAT WOULD DISTINGUISH THEM

  H1  UNSUPERVISED EXTRAPOLATION
      The adapter edits every vertex but the loss only constrains the ones the
      training camera saw. Vertices it never saw get an unverified edit.
      PREDICTION: |delta| is substantially larger on invisible vertices than on
      visible ones. If |delta| is the SAME on both, H1 is dead.

  H2  THE ADAPTER LEARNED A FRONT-VIEW PROJECTION, i.e. colour ~ f(height)
      From one camera, vertex height maps almost directly to image row, so the
      cheapest function fitting the front view is one of height. Applied to the
      whole object that paints horizontal bands.
      PREDICTION: delta regressed on the vertical axis has high R^2, and markedly
      higher than the frozen colour's own R^2 against the same axis (the control —
      real lava has some vertical structure too, so the baseline matters).
      If R^2(delta) is no higher than R^2(frozen colour), H2 is dead.

  H3  DEPTH / VIEW-DEPENDENCE
      The edit tracks distance from the training camera rather than height.
      PREDICTION: R^2 against the camera-axis coordinate beats R^2 against height.

  These are not mutually exclusive. The numbers say how much of each.

VISIBILITY IS MEASURED, NOT APPROXIMATED
  A vertex is "seen by the training camera" iff its colour influenced the rendered
  image — that is, iff d(sum of rendered pixels)/d(that vertex's colour) != 0.
  Computed by autograd through the real rasterizer, so occlusion, backfacing and
  clipping are all handled exactly by nvdiffrast rather than by a projection
  formula I would have to get right myself.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/measure_sticker.py
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
ap.add_argument('--run',     default='rung5_colonly_r4_s6_5d6b1700')
ap.add_argument('--ckpt',    default='lora_best.pt')
ap.add_argument('--frames',  default='1,75,150')
ap.add_argument('--abs-min', type=float, default=-9.0)
ap.add_argument('--abs-max', type=float, default=8.0)
ap.add_argument('--out-dir', default=None, type=Path)
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'sticker_analysis')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_DIM = 768
COLOR_START, COLOR_END = 53, 101


def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p = torch.cuda.get_device_properties(0)
    arch_tag, arch_str = f'sm{p.major}{p.minor}', f'{p.major}.{p.minor}'
    local = f'/tmp/nvdiffrast_{arch_tag}'
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
             '--no-cache-dir', '--no-deps', '-q'], cwd=f'{src}/nvdiffrast', env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS)

DEVICE = torch.device('cuda')


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class DecBlockLoRABundle(nn.Module):
    def __init__(self, rank, dim=DEC_DIM):
        super().__init__()
        mlp_h = dim * 4
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
            d = _lb.lora_fc1(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))

        def _fc2_hook(mod, inp, out, _lb=lb):
            d = _lb.lora_fc2(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))

        handles.append(block.attn.to_qkv.register_forward_hook(_qkv_hook))
        handles.append(block.attn.to_out.register_forward_hook(_out_hook))
        handles.append(block.mlp.mlp[0].register_forward_hook(_fc1_hook))
        handles.append(block.mlp.mlp[2].register_forward_hook(_fc2_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def two_meshes(dec_model, registry, slat):
    """Frozen mesh and colonly mesh from the same SLaT. Identical geometry."""
    captured = {}

    def _hook(key):
        def _fn(mod, inp, out):
            captured[key] = out
        return _fn

    with torch.no_grad():
        h1 = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        frozen_meshes = dec_model(slat)
        h1.remove()
        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_block_lora_ctx(dec_model, registry):
            dec_model(slat)
        h2.remove()
        fz, lo = captured['frozen'], captured['lora']
        geom = fz.feats[:, :COLOR_START]
        col  = lo.feats[:, COLOR_START:COLOR_END].clamp(args.abs_min, args.abs_max)
        colonly_meshes = dec_model.to_representation(
            fz.replace(torch.cat([geom, col], dim=1)))
    return frozen_meshes[0], colonly_meshes[0]


def visible_vertex_mask(mesh, renderer):
    """
    EXACT visibility: a vertex is seen iff its colour affected the rendered image.
    d(sum of pixels)/d(vertex colour) != 0. nvdiffrast handles occlusion,
    backfacing and clipping; no projection formula of mine is involved.
    """
    attrs = mesh.vertex_attrs.detach().clone().requires_grad_(True)
    saved = mesh.vertex_attrs
    mesh.vertex_attrs = attrs
    res = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color'])
    res['color'].sum().backward()
    g = attrs.grad[:, :3].abs().sum(dim=1)
    mesh.vertex_attrs = saved
    return (g > 0)


def r2(y, x):
    """R^2 of a least-squares fit y ~ a*x + b. Both 1-D."""
    x = x.double(); y = y.double()
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return 0.0
    b = ((x - xm) * (y - ym)).sum() / sxx
    pred = b * (x - xm) + ym
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    frames = [int(x) for x in args.frames.split(',')]
    cfg = json.load(open(RUN_DIR / 'config.json'))
    ck  = torch.load(CKPT, map_location='cpu', weights_only=True)
    print('=' * 94)
    print('STICKER ANALYSIS — per-vertex, rung5_colonly')
    print(f'  ckpt epoch={ck["epoch"]}  blocks={cfg["active_blocks"]}  rank={cfg["rank"]}')
    print(f'  frames {frames}')
    print('=' * 94, flush=True)

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()

    pipe = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec  = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    for n in list(pipe.models.keys()):
        if n != 'slat_decoder_mesh':
            try: pipe.models[n].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    registry = DecLoRARegistry(cfg['active_blocks'], cfg['rank']).to(DEVICE)
    registry.load_state_dict(ck['registry_state'], strict=True)
    registry.eval()
    renderer = make_renderer(DEVICE)

    rows = []
    for fi in frames:
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))
        m_f, m_c = two_meshes(dec, registry, st)
        assert m_f.vertices.shape[0] == m_c.vertices.shape[0], 'geometry differs'

        V   = m_f.vertices.detach()                       # (N,3)
        cf  = m_f.vertex_attrs[:, :3].detach().float()    # frozen RGB
        cc  = m_c.vertex_attrs[:, :3].detach().float()    # colonly RGB
        d   = (cc - cf)                                   # the whole artifact
        dm  = d.abs().mean(dim=1)                         # per-vertex |delta|

        vis = visible_vertex_mask(m_f, renderer)
        nv, nvis = V.shape[0], int(vis.sum())

        # H1 — is the edit bigger where the camera never looked?
        dv, di = dm[vis], dm[~vis]

        # H2 / H3 — what is the delta a function of?
        #   the camera sits at (0,-r,0) looking +y, world_up = (0,0,-1)
        #   so world z is the vertical axis and world y is the camera axis
        rr = {ax: r2(dm, V[:, k]) for k, ax in enumerate(('x', 'y_cameraaxis', 'z_vertical'))}
        # control: the frozen colour's own structure along the same axes
        cfm = cf.mean(dim=1)
        rc = {ax: r2(cfm, V[:, k]) for k, ax in enumerate(('x', 'y_cameraaxis', 'z_vertical'))}

        row = dict(frame=fi, n_verts=nv, n_visible=nvis,
                   frac_visible=nvis / nv,
                   delta_mean_visible=float(dv.mean()), delta_mean_invisible=float(di.mean()),
                   delta_std_visible=float(dv.std()), delta_std_invisible=float(di.std()),
                   ratio_invis_vis=float(di.mean() / max(dv.mean(), 1e-9)),
                   r2_delta=rr, r2_frozen_colour=rc)
        rows.append(row)

        print(f'\n[frame {fi}]  verts={nv:,}   visible from training camera='
              f'{nvis:,} ({nvis/nv*100:.1f}%)')
        print(f'  H1  |delta| on VISIBLE   vertices = {dv.mean():.4f} +- {dv.std():.4f}')
        print(f'      |delta| on INVISIBLE vertices = {di.mean():.4f} +- {di.std():.4f}')
        print(f'      ratio invisible/visible       = {di.mean()/max(dv.mean(),1e-9):.3f}'
              f'   {"<- H1 SUPPORTED" if di.mean() > 1.3*dv.mean() else "<- H1 NOT supported"}')
        print(f'  H2/H3  R^2 of |delta| against position:')
        for ax in rr:
            print(f'         {ax:14s} delta R^2={rr[ax]:.4f}   '
                  f'(frozen colour R^2={rc[ax]:.4f})')
        del st, m_f, m_c
        gc.collect(); torch.cuda.empty_cache()

    print('\n' + '=' * 94)
    print(f'{"frame":>6}{"verts":>10}{"vis%":>8}{"|d|vis":>10}{"|d|invis":>10}'
          f'{"ratio":>8}{"R2 vert":>10}{"R2 camax":>10}')
    print('-' * 94)
    for r in rows:
        print(f'{r["frame"]:>6}{r["n_verts"]:>10,}{r["frac_visible"]*100:>7.1f}%'
              f'{r["delta_mean_visible"]:>10.4f}{r["delta_mean_invisible"]:>10.4f}'
              f'{r["ratio_invis_vis"]:>8.3f}{r["r2_delta"]["z_vertical"]:>10.4f}'
              f'{r["r2_delta"]["y_cameraaxis"]:>10.4f}')
    print('=' * 94, flush=True)

    json.dump({'run': args.run, 'ckpt_epoch': ck['epoch'], 'rows': rows},
              open(OUT_DIR / 'sticker_analysis.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "sticker_analysis.json"}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
