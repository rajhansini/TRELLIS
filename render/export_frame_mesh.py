"""
export_frame_mesh.py
--------------------
Decode one frame of a trained run and write it out as .glb and .ply so the
Blender pipeline can render it.

Writes, per frame, for each of two arms:
    frozen  — the unmodified TRELLIS decoder            (the baseline)
    result  — the trained adapter's output               (ours)

Both arms use the SAME geometry at a given frame (that is the two-pass splice's
guarantee, verified 0/150 frames elsewhere), so the only difference between the
two files is per-vertex colour.

OUTPUT FORMATS
  .glb  the exchange format, vertex colours embedded
  .ply  what render_single_mesh.py can actually import

  Both are written because the point of the exercise is to exercise
  glb -> ply -> Blender end to end. If you only need the render, the .ply is
  enough; glb_to_ply.py exists for assets that arrive as .glb from elsewhere.

COLOUR
  TRELLIS meshes carry colour as a per-vertex attribute — mesh.vertex_attrs[:, :3]
  — with no UVs anywhere. So the export is exact; nothing is baked or resampled.

ALIGNMENT
  With --alignment, vertices get the solved similarity transform applied before
  export, i.e. the asset is exactly what the trained run renders. It is a rigid
  transform plus uniform scale, so it does not change how the object looks from a
  free camera in Blender — but it keeps the exported asset consistent with every
  number we report.

Usage:
  python render/export_frame_mesh.py --frame 75 \
      --run rung13_aligned_r4_s6_c742c0b3 \
      --alignment experiments/lora_experiments/visibility/alignment/alignment.json \
      --out-dir render/out/f0075
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
_ROOT = _HERE.parent
_LORA = _ROOT / 'experiments' / 'lora_experiments'
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--run',       default='rung13_aligned_r4_s6_c742c0b3')
ap.add_argument('--ckpt',      default='lora_best.pt')
ap.add_argument('--frame',     type=int, default=75)
ap.add_argument('--frames',    default=None,
                help='range like 1-150 for batch export; overrides --frame')
ap.add_argument('--alignment', default=None)
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--abs-min',   type=float, default=-9.0)
ap.add_argument('--abs-max',   type=float, default=8.0)
ap.add_argument('--no-glb',    action='store_true')
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'out' / f'f{args.frame:04d}')).resolve()
FRAMES = ([int(x) for x in range(int(args.frames.split('-')[0]),
                               int(args.frames.split('-')[1]) + 1)]
          if args.frames else [args.frame])
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
import trimesh

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

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


def two_meshes(dec_model, registry, slat):
    """Frozen mesh and adapted mesh from one SLaT. Identical geometry by construction."""
    cap = {}

    def _hook(key):
        def _fn(mod, inp, out):
            cap[key] = out
        return _fn

    with torch.no_grad():
        h1 = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        frozen_meshes = dec_model(slat)
        h1.remove()
        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_block_lora_ctx(dec_model, registry):
            dec_model(slat)
        h2.remove()
        fz, lo = cap['frozen'], cap['lora']
        geom = fz.feats[:, :COLOR_START]
        col  = lo.feats[:, COLOR_START:COLOR_END].clamp(args.abs_min, args.abs_max)
        adapted = dec_model.to_representation(fz.replace(torch.cat([geom, col], 1)))[0]
    return frozen_meshes[0], adapted


def rodrigues(rv):
    rv = np.asarray(rv, float); th = float(np.linalg.norm(rv)) + 1e-12
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def export(mesh, tag, align, verbose=True):
    v = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    f = mesh.faces.detach().cpu().numpy().astype(np.int64)
    c = mesh.vertex_attrs[:, :3].detach().float().cpu().numpy()

    if align is not None:
        v = align['s'] * ((v - align['c']) @ align['R'].T) + align['c'] + align['t']

    lo, hi = float(c.min()), float(c.max())
    rgba = np.concatenate(
        [(np.clip(c, 0, 1) * 255).astype(np.uint8),
         np.full((len(c), 1), 255, np.uint8)], axis=1)

    m = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgba, process=False)
    ply = OUT_DIR / f'{tag}.ply'
    m.export(str(ply))
    outs = [ply]
    if not args.no_glb:
        glb = OUT_DIR / f'{tag}.glb'
        m.export(str(glb))
        outs.append(glb)

    b = m.bounds
    if not verbose:
        return m
    print(f'  [{tag}]  verts={len(v):,}  faces={len(f):,}')
    print(f'      raw colour range [{lo:.3f}, {hi:.3f}]'
          + ('  <-- outside [0,1], clipped' if lo < -1e-3 or hi > 1 + 1e-3 else ''))
    print(f'      mean RGB {np.round(rgba[:, :3].mean(axis=0), 1).tolist()}')
    print(f'      bounds   min {np.round(b[0], 4).tolist()}  max {np.round(b[1], 4).tolist()}')
    for o in outs:
        print(f'      wrote    {o.name}  ({o.stat().st_size/1e6:.1f} MB)')
    return m


def main():
    assert CKPT.exists(),     f'missing {CKPT}'
    assert SLAT_NPZ.exists(), f'missing {SLAT_NPZ}'
    cfg = json.load(open(RUN_DIR / 'config.json'))
    ck  = torch.load(CKPT, map_location='cpu', weights_only=True)
    print('=' * 78)
    print(f'EXPORT frames {FRAMES[0]}..{FRAMES[-1]} ({len(FRAMES)})  from  {args.run}')
    print(f'  ckpt epoch {ck["epoch"]}   blocks {cfg["active_blocks"]}   rank {cfg["rank"]}')

    align = None
    if args.alignment:
        a = json.load(open(args.alignment))
        align = dict(s=float(a['scale']), R=rodrigues(a['rotvec']),
                     c=np.asarray(a['centre'], float),
                     t=np.asarray(a['translation'], float))
        print(f'  alignment  s={align["s"]:.4f}  t={np.round(align["t"],4).tolist()}')
    else:
        print('  alignment  NONE (exporting raw decoder output)')
    print('=' * 78, flush=True)

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
    print(f'[LORA] {sum(p.numel() for p in registry.parameters()):,} params '
          f'(strict load OK)\n', flush=True)

    verbose = len(FRAMES) == 1
    mismatch = 0
    for k, fi in enumerate(FRAMES, 1):
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))
        m_frozen, m_result = two_meshes(dec, registry, st)
        if m_frozen.vertices.shape[0] != m_result.vertices.shape[0]:
            mismatch += 1
        export(m_frozen, f'frozen_f{fi:04d}', align, verbose)
        export(m_result, f'result_f{fi:04d}', align, verbose)
        if not verbose and (k % 10 == 0 or k == 1 or k == len(FRAMES)):
            print(f'  {k:3d}/{len(FRAMES)}  f{fi:04d}  '
                  f'verts={m_frozen.vertices.shape[0]:,}', flush=True)
        del st, m_frozen, m_result
        gc.collect(); torch.cuda.empty_cache()
    print(f'\n[CHECK] frames where the two arms differed in vertex count: '
          f'{mismatch}/{len(FRAMES)}')
    assert mismatch == 0, 'the splice is not holding'

    print(f'\n[SAVE] {OUT_DIR}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
