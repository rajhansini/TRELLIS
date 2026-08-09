"""
render_visibility_overlay.py
----------------------------
Does the white artifact sit exactly on the part the camera never saw?

TWO COMPETING READS OF THE SAME PICTURE
  A  "a texture is being pasted onto the mesh in the wrong place"
  B  "the white is the set of vertices the training camera never saw, so its
      boundary is a hard, geometry-ignoring edge that LOOKS like a pasted edge"

  These predict different things and one picture separates them:

    if B  the white region and the never-seen region coincide
    if A  white lands on vertices the camera DID see, i.e. somewhere the loss
          actually graded, which no visibility story can explain

PANELS
  1  colonly render                       what you are looking at
  2  per-vertex visibility, painted       white = seen by the training camera
                                          black = never seen, not once
  3  where colonly renders near-white     the artifact, isolated
  4  agreement map                        red   = white artifact on a SEEN vertex
                                                  (supports A)
                                          green = white artifact on an UNSEEN one
                                                  (supports B)

THE NUMBER
  of the pixels where colonly renders near-white, what fraction is backed by
  vertices with zero visibility. Near 100% -> B. Near 0% -> A.

VISIBILITY IS MEASURED, NOT PROJECTED
  Same definition as measure_sticker.py: a vertex is seen iff
  d(sum of rendered pixels)/d(that vertex's colour) != 0, via autograd through
  the real rasteriser. Occlusion, backfacing and clipping are nvdiffrast's
  answer. Computed from the TRAINING camera (EXTRINSICS), then the result is
  viewed from wherever --theta says.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_visibility_overlay.py \
    --frames 1,75 --thetas 0,328
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
ap.add_argument('--run',       default='rung5_colonly_r4_s6_5d6b1700')
ap.add_argument('--ckpt',      default='lora_best.pt')
ap.add_argument('--frames',    default='1,75')
ap.add_argument('--thetas',    default='0,328')
ap.add_argument('--elevation', type=float, default=15.0)
ap.add_argument('--radius',    type=float, default=2.0)
ap.add_argument('--white-thr', type=float, default=235.0,
                help='a pixel counts as the white artifact if min(RGB) exceeds this')
ap.add_argument('--abs-min',   type=float, default=-9.0)
ap.add_argument('--abs-max',   type=float, default=8.0)
ap.add_argument('--out-dir',   default=None, type=Path)
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'visibility_overlay')).resolve()
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
from PIL import Image, ImageDraw

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


def colonly_mesh(dec_model, registry, slat):
    captured = {}

    def _hook(key):
        def _fn(mod, inp, out):
            captured[key] = out
        return _fn

    with torch.no_grad():
        h1 = dec_model.out_layer.register_forward_hook(_hook('frozen'))
        dec_model(slat)
        h1.remove()
        h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
        with dec_block_lora_ctx(dec_model, registry):
            dec_model(slat)
        h2.remove()
        fz, lo = captured['frozen'], captured['lora']
        geom = fz.feats[:, :COLOR_START]
        col  = lo.feats[:, COLOR_START:COLOR_END].clamp(args.abs_min, args.abs_max)
        return dec_model.to_representation(fz.replace(torch.cat([geom, col], 1)))[0]


def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    theta, el = math.radians(theta_deg), math.radians(elevation_deg)
    C = torch.tensor([radius * math.sin(theta) * math.cos(el),
                      -radius * math.cos(theta) * math.cos(el),
                      radius * math.sin(el)], dtype=torch.float32)
    world_up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cam_z = -C / C.norm()
    cam_x = torch.linalg.cross(world_up, cam_z); cam_x = cam_x / cam_x.norm()
    cam_y = torch.linalg.cross(cam_z, cam_x);    cam_y = cam_y / cam_y.norm()
    R = torch.stack([cam_x, cam_y, cam_z], dim=0)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3,  3] = R @ (-C)
    return E.to(DEVICE)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render_rgb(mesh, renderer, ext):
    res = renderer.render(mesh, ext, INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
    m = res['mask']
    c = (res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))).detach().clamp(0, 1)
    return (c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), (m > 0.5)


def vertex_visibility(mesh, renderer):
    """Seen iff d(rendered pixels)/d(vertex colour) != 0, from the TRAINING camera."""
    attrs = mesh.vertex_attrs.detach().clone().requires_grad_(True)
    saved = mesh.vertex_attrs
    mesh.vertex_attrs = attrs
    res = renderer.render(mesh, EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color'])
    res['color'].sum().backward()
    g = attrs.grad[:, :3].abs().sum(dim=1)
    mesh.vertex_attrs = saved
    return (g > 0)


def paint(mesh, rgb_per_vertex):
    """Return a shallow copy of mesh whose albedo is rgb_per_vertex (N,3) in [0,1]."""
    import copy
    m2 = copy.copy(mesh)
    a  = mesh.vertex_attrs.detach().clone()
    a[:, :3] = rgb_per_vertex
    if a.shape[1] > 3:
        a[:, 3:] = 0.0
    m2.vertex_attrs = a
    return m2


LAB = 30
def strip(imgs, labels):
    H, W = imgs[0].shape[:2]
    pil = Image.fromarray(np.full((H + LAB, W * len(imgs), 3), 18, np.uint8))
    dr  = ImageDraw.Draw(pil)
    for i, (a, l) in enumerate(zip(imgs, labels)):
        pil.paste(Image.fromarray(a), (i * W, LAB))
        dr.rectangle([i * W, 0, (i + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
        dr.text((i * W + 8, 9), l, fill=(240, 240, 240))
    return pil


def main():
    cfg = json.load(open(RUN_DIR / 'config.json'))
    ck  = torch.load(CKPT, map_location='cpu', weights_only=True)
    frames = [int(x) for x in args.frames.split(',')]
    thetas = [float(x) for x in args.thetas.split(',')]
    print(f'[CKPT] epoch={ck["epoch"]}  frames={frames}  thetas={thetas}', flush=True)

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
        mesh = colonly_mesh(dec, registry, st)
        vis  = vertex_visibility(mesh, renderer)          # from the TRAINING camera
        nvis = int(vis.sum()); nv = vis.numel()
        print(f'\n[frame {fi}]  verts={nv:,}  seen by training camera='
              f'{nvis:,} ({nvis/nv*100:.1f}%)', flush=True)

        vis_rgb = vis.float().unsqueeze(1).repeat(1, 3)   # white=seen, black=unseen
        m_vis   = paint(mesh, vis_rgb)

        for th in thetas:
            ext = orbit_extrinsics(th, args.elevation, args.radius)
            rgb_c, mask_c = render_rgb(filter_degenerate_faces(mesh), renderer, ext)
            rgb_v, _      = render_rgb(filter_degenerate_faces(m_vis), renderer, ext)

            mc = mask_c.cpu().numpy()
            white = (rgb_c.min(axis=2) > args.white_thr) & mc      # the artifact
            seen  = (rgb_v[..., 0] > 128) & mc                     # visible region

            art = np.full_like(rgb_c, 255)
            art[white] = (255, 60, 60)
            agree = np.full_like(rgb_c, 255)
            agree[white & seen]  = (220, 40, 40)     # white on a SEEN vertex  -> A
            agree[white & ~seen] = (40, 190, 90)     # white on an UNSEEN one  -> B

            n_white = int(white.sum())
            on_unseen = int((white & ~seen).sum())
            frac = on_unseen / max(n_white, 1)
            rows.append(dict(frame=fi, theta=th, n_white=n_white,
                             frac_white_on_unseen=frac,
                             frac_object_white=n_white / max(int(mc.sum()), 1)))
            print(f'   theta={th:6.1f}  white px={n_white:7,} '
                  f'({n_white/max(int(mc.sum()),1)*100:5.1f}% of object)   '
                  f'of those, on NEVER-SEEN vertices = {frac*100:6.2f}%', flush=True)

            strip([rgb_c, rgb_v, art, agree],
                  [f'colonly  f{fi:04d}  theta={th:.0f}',
                   'visibility  white=seen / black=never seen',
                   f'white artifact  ({n_white/max(int(mc.sum()),1)*100:.1f}% of object)',
                   f'red=white on SEEN   green=white on UNSEEN  ({frac*100:.1f}% green)']
                  ).save(OUT_DIR / f'overlay_f{fi:04d}_t{int(th):03d}.png')

        del st, mesh, m_vis
        gc.collect(); torch.cuda.empty_cache()

    print('\n' + '=' * 84)
    print(f'{"frame":>6}{"theta":>8}{"white px":>11}{"% of obj":>10}{"white on UNSEEN":>18}')
    print('-' * 84)
    for r in rows:
        print(f'{r["frame"]:>6}{r["theta"]:>8.1f}{r["n_white"]:>11,}'
              f'{r["frac_object_white"]*100:>9.1f}%{r["frac_white_on_unseen"]*100:>17.2f}%')
    print('=' * 84)
    print('near 100% -> the white IS the never-seen set (unsupervised extrapolation)')
    print('near   0% -> the white sits where the loss DID look; a visibility story '
          'cannot explain it')
    json.dump(rows, open(OUT_DIR / 'visibility_overlay.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
