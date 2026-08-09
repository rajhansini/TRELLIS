"""
measure_sdf_sheets.py
---------------------
Where does the second teapot come from?

CLAIM UNDER TEST
  The pale outer teapot is a SECOND ISOSURFACE. FlexiCubes draws the surface
  wherever the sdf field changes sign. A ray fired through a solid teapot should
  cross zero exactly twice: once entering, once leaving. If a LoRA perturbs sdf
  enough to create spurious extra zeros, the same ray crosses FOUR times and
  FlexiCubes emits an extra shell outside the real body.

  Frozen decoder  ──►  2 crossings per ray   (one solid surface)
  LoRA arm        ──►  4+ crossings per ray  (nested shells)  <- the prediction

  This does not touch the renderer. It reads the sdf field itself, so the answer
  is about the geometry, not about shading or transparency.

METHOD  (mirrors cube2mesh.py exactly, no reimplementation)
  hook out_layer -> feats (N_fine, 101)
  sdf   = feats[:, 0:8].reshape(-1, 8, 1)          layouts from cube2mesh
  sdf  += sdf_bias = -1/res
  sparse_cube2verts(coords, sdf)                   the real averaging over shared corners
  get_dense_attrs(v_pos, ..., res+1, sdf_init=True)  the real 257^3 scatter
  -> sdf_d (257, 257, 257)

  then, along every grid line in x, y and z:
      count sign changes, ignoring the constant +1 exterior

REPORTED
  crossings per ray, as a histogram    2 = one shell, 4+ = nested shells
  total crossings                      absolute amount of surface
  frac_rays_gt2                        THE number: fraction of rays hitting >2
  |sdf| distribution near zero         how close the field sits to flipping
  con_loss                             neighbouring cubes disagreeing (never
                                       computed in any run — it is behind
                                       `if training:` and everything runs eval)

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/measure_sdf_sheets.py
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
ap.add_argument('--frames',  default='1,75,150')
ap.add_argument('--out-dir', default=None, type=Path)
args = ap.parse_args()

SLAT_NPZ = _LORA / 'runs' / 'rung5_colonly_outlayer_r4_s6_c85c888f' / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'sdf_sheets')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_DIM = 768
COLOR_START, COLOR_END = 53, 101

# (label, run dir, kind)  kind: 'block' = registry hooks, 'outlayer' = out_layer LoRA
ARMS = [
    ('rung5.4  blk 0-11 r4  DIRECT', _LORA / 'runs' / 'rung5_54_r4_s6_31e5b3e8',  'block'),
    ('rung5.5  blk 8-11 r16 DIRECT', _LORA / 'runs' / 'rung5_55_r16_s6_9c47bd2e', 'block'),
    ('rung5colonly blk 0-11 r4',     _LORA / 'runs' / 'rung5_colonly_r4_s6_5d6b1700', 'block'),
    ('rung9    out_layer+geom',      _HERE / 'runs' / 'rung9_geom-color_r4_s6_e44d8696', 'outlayer'),
]


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
# the REAL functions cube2mesh uses — not reimplementations
from trellis.representations.mesh.utils_cube import sparse_cube2verts, get_dense_attrs

DEVICE = torch.device('cuda')


# ── LoRA structures, verbatim from rung5_subladder.py ────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.scaling = 1.0
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

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
    def __init__(self, rank, in_dim, out_dim):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def _clamp_colour_only(rw, s, e):
    lo, hi = max(s, COLOR_START), min(e, COLOR_END)
    if lo >= hi:
        return rw
    if lo == s and hi == e:
        return rw.clamp(-9.0, 8.0)
    out = rw.clone()
    out[:, lo - s:hi - s] = rw[:, lo - s:hi - s].clamp(-9.0, 8.0)
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


# ── the measurement ──────────────────────────────────────────────────────────

def capture_outlayer(dec_model, slat, ctx=None):
    """Run the decoder, return out_layer's (N_fine, 101) output. Mesh discarded."""
    got = {}

    def _h(mod, inp, out):
        got['f'] = out.feats.detach()
    with torch.no_grad():
        hd = dec_model.out_layer.register_forward_hook(_h)
        if ctx is None:
            dec_model(slat)
        else:
            with ctx():
                dec_model(slat)
        hd.remove()
    return got['f']


def dense_sdf(feats, coords, mesh_res):
    """
    Reproduce cube2mesh's sdf path exactly, using its own functions.
      feats  (N_fine, 101)   out_layer output
      coords (N_fine, 3)     integer cube coords
    Returns (res+1)^3 float32 sdf grid on GPU.
    """
    sdf = feats[:, 0:8].reshape(-1, 8, 1).float()      # layouts: sdf = (8,1)
    sdf = sdf + (-1.0 / mesh_res)                      # cube2mesh: self.sdf_bias
    v_pos, v_attrs, con = sparse_cube2verts(coords, sdf, training=True)
    d = get_dense_attrs(v_pos, v_attrs, res=mesh_res + 1, sdf_init=True)
    return d[..., 0].reshape(mesh_res + 1, mesh_res + 1, mesh_res + 1), float(con)


def crossing_stats(sdf_d):
    """
    Count sign changes along every grid line, in all three axes.
    A solid object gives 2 per ray that hits it. 4+ means nested shells.
    """
    s = torch.sign(sdf_d)
    out = {}
    per_ray_all = []
    total = 0
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        flips = (s.diff(dim=ax) != 0)
        n = flips.sum(dim=ax)                    # crossings per ray
        per_ray_all.append(n.flatten())
        total += int(flips.sum().item())
    allr = torch.cat(per_ray_all)
    hit  = allr[allr > 0]                        # rays that touch the object
    hist = {}
    for k in (1, 2, 3, 4, 5, 6):
        hist[k] = int((hit == k).sum().item())
    hist['7+'] = int((hit >= 7).sum().item())
    out['total_crossings'] = total
    out['rays_hit']        = int(hit.numel())
    out['hist']            = hist
    out['frac_rays_gt2']   = float((hit > 2).sum().item()) / max(int(hit.numel()), 1)
    out['mean_per_ray']    = float(hit.float().mean().item()) if hit.numel() else 0.0
    interior = sdf_d[sdf_d.abs() < 0.5]
    out['frac_abs_lt_0p01'] = float((interior.abs() < 0.01).sum().item()) / max(interior.numel(), 1)
    return out


def build_arm(run_dir, kind, dec_model):
    cfg = json.load(open(run_dir / 'config.json'))
    ck  = torch.load(run_dir / 'lora_ckpts' / 'lora_best.pt',
                     map_location='cpu', weights_only=True)
    if kind == 'block':
        reg = DecLoRARegistry(cfg['active_blocks'], cfg['rank']).to(DEVICE)
        reg.load_state_dict(ck['registry_state'], strict=True)
        reg.eval()
        return (lambda: dec_block_lora_ctx(dec_model, reg),
                f"blocks {cfg['active_blocks']} r{cfg['rank']} ep{ck['epoch']}")
    blocks = [tuple(b) for b in ck['blocks']] if 'blocks' in ck else [(53, 101)]
    offs, o = [], 0
    for s, e in blocks:
        offs.append(o); o += e - s
    Bw = ck['lora_state']['B']
    assert Bw.shape[0] == o
    lora = OutLayerLoRA(Bw.shape[1], 96, o).to(DEVICE)
    lora.load_state_dict(ck['lora_state'], strict=True)
    lora.eval()
    return (lambda: out_layer_lora_ctx(dec_model, lora, blocks, offs),
            f"{blocks} r{Bw.shape[1]} ep{ck['epoch']}")


def main():
    frames = [int(x) for x in args.frames.split(',')]
    print('=' * 96)
    print('SDF SHEET COUNT — is the pale outer teapot a second isosurface?')
    print(f'  frames: {frames}')
    print('  a solid object crosses zero TWICE per ray. 4+ means nested shells.')
    print('=' * 96, flush=True)

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

    MESH_RES = dec.mesh_extractor.res
    print(f'[RES] mesh_extractor.res = {MESH_RES}  -> grid {MESH_RES+1}^3', flush=True)

    def slat(fi):
        return sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))

    rows = []

    def run(label, ctx, note):
        agg = {'total_crossings': [], 'rays_hit': [], 'frac_rays_gt2': [],
               'mean_per_ray': [], 'frac_abs_lt_0p01': [], 'con_loss': []}
        hists = []
        for fi in frames:
            st = slat(fi)
            feats = capture_outlayer(dec, st, ctx)
            fine_coords = None
            # out_layer output is per FINE voxel; recover their coords by re-running
            # the decoder body is unnecessary — the hook's module input carries them.
            # Instead capture coords alongside feats:
            g = {}
            def _h2(mod, inp, out):
                g['c'] = out.coords[:, 1:].detach()
            with torch.no_grad():
                hd = dec.out_layer.register_forward_hook(_h2)
                if ctx is None: dec(st)
                else:
                    with ctx(): dec(st)
                hd.remove()
            fine_coords = g['c']
            sdf_d, con = dense_sdf(feats, fine_coords, MESH_RES)
            s = crossing_stats(sdf_d)
            for k in agg:
                agg[k].append(s[k] if k in s else con)
            hists.append(s['hist'])
            del st, feats, sdf_d, fine_coords
            gc.collect(); torch.cuda.empty_cache()
        H = {k: int(np.mean([h[k] for h in hists])) for k in hists[0]}
        row = {'name': label, 'note': note,
               **{k: float(np.mean(v)) for k, v in agg.items()}, 'hist': H}
        rows.append(row)
        print(f'\n[{label}]  {note}')
        print(f'   crossings/ray (mean over rays that hit) = {row["mean_per_ray"]:.3f}')
        print(f'   FRAC OF RAYS WITH >2 CROSSINGS          = {row["frac_rays_gt2"]*100:.2f}%')
        print(f'   total crossings = {row["total_crossings"]:,.0f}   rays hit = {row["rays_hit"]:,.0f}')
        print(f'   histogram {H}')
        print(f'   con_loss (cube disagreement) = {row["con_loss"]:.6f}', flush=True)

    run('frozen decoder', None, 'no adapter')
    for label, run_dir, kind in ARMS:
        if not (run_dir / 'lora_ckpts' / 'lora_best.pt').exists():
            print(f'  [SKIP] {label}'); continue
        ctx, note = build_arm(run_dir, kind, dec)
        run(label, ctx, note)

    print('\n' + '=' * 96)
    print(f'{"arm":34s}{"cross/ray":>11s}{"rays>2":>10s}{"total":>13s}{"con_loss":>12s}')
    print('-' * 96)
    for r in rows:
        print(f'{r["name"]:34s}{r["mean_per_ray"]:>11.3f}{r["frac_rays_gt2"]*100:>9.2f}%'
              f'{r["total_crossings"]:>13,.0f}{r["con_loss"]:>12.6f}')
    print('=' * 96)
    print('rays>2 is the verdict: a single closed surface gives 2 crossings per ray.')
    print('Anything above that is extra sheets — the second teapot.')
    json.dump({'frames': frames, 'mesh_res': MESH_RES, 'rows': rows},
              open(OUT_DIR / 'sdf_sheets.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "sdf_sheets.json"}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
