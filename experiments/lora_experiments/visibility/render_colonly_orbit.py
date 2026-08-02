"""
render_colonly_orbit.py
-----------------------
360 degree turntable of rung5_colonly:  GT | frozen TRELLIS | colonly.

WHY A NEW SCRIPT
  render_vismask_orbit.py is out_layer-specific — it multiplies a per-voxel colour
  delta by a visibility weight, which only exists when the adapter is a pointwise
  map at out_layer. rung5_colonly's adapter lives inside the 12 decoder blocks, so
  there is no per-voxel delta to weight. This reuses the verified two-pass
  colonly_forward instead.

THE GT PANEL
  The GT is a single-camera video. There is no ground truth at any other azimuth.
  At theta=0 the GT panel is the real frame; at every other angle it is that same
  front-view frame, shown for reference only and labelled as such. Do not read it
  as a target away from 0 degrees.

WHAT THE TURNTABLE ANSWERS
  Whether the learned lava wraps the whole object or is pasted onto the side the
  training camera saw. The frozen panel is the control: it already wraps lava
  correctly all round, so if colonly matches it everywhere except in colour
  detail, the edit is uniform. If colonly is correct at 0 degrees and degrades as
  the camera turns, the adapter learned a sticker.

  Geometry is spliced from the frozen pass, so the two right-hand silhouettes
  must be identical at every angle. The labels carry vertex counts; the script
  asserts nothing but reports every mismatch.

MODES
  --sweep angle : SLaT frame fixed, camera orbits 360    (isolates the sticker)
  --sweep both  : frame advances 1..150 while orbiting   (the "360 video")

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_colonly_orbit.py \
    --sweep angle --frame 75 --n-angles 90 --elevation 15
"""

import sys, os, gc, json, math, argparse, subprocess
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
ap.add_argument('--sweep',     default='angle', choices=['angle', 'both'])
ap.add_argument('--frame',     type=int, default=75, help='SLaT frame for --sweep angle')
ap.add_argument('--n-angles',  type=int, default=90)
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--elevation', type=float, default=15.0)
ap.add_argument('--radius',    type=float, default=2.0)
ap.add_argument('--fps',       type=int, default=15)
ap.add_argument('--abs-min',   type=float, default=-9.0)
ap.add_argument('--abs-max',   type=float, default=8.0)
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--vismask',   default=None,
                help="visibility.npz -> reproduce rung11's masked forward")
ap.add_argument('--mask-mode', default='soft', choices=['none','hard','soft'])
ap.add_argument('--mask-q',    type=float, default=0.50)
ap.add_argument('--mask-gamma',type=float, default=1.0)
ap.add_argument('--mask-eps',  type=float, default=1e-4)
ap.add_argument('--alignment', default=None,
                help="alignment.json -> reproduce rung13's aligned render")
ap.add_argument('--panel-name', default='rung5_colonly')
ap.add_argument('--no-gt',     action='store_true',
                help='drop the GT panel. The GT is a single-camera 2D video, so it\n'
                     'cannot orbit — in a turntable it is a static distraction.')
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
GT_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                '/outputs/teapot_lava_kling_premium'
                '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (RUN_DIR / 'orbit' /
            (f'sweep_{args.sweep}' + ('_nogt' if '--no-gt' in sys.argv else '')))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_DIM = 768
COLOR_START, COLOR_END = 53, 101


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

# ── variant switches: reproduce rung11 (visibility mask) or rung13 (alignment)
W_VOX = None
if args.vismask:
    sys.path.insert(0, str(_HERE))
    from vis_mask import VisibilityMask
    _vm = VisibilityMask(args.vismask)
    print(f'[VISMASK] {_vm.describe(mode=args.mask_mode, soft_q=args.mask_q, gamma=args.mask_gamma, eps=args.mask_eps)}', flush=True)
    W_VOX = _vm.weights_torch(DEVICE, mode=args.mask_mode, soft_q=args.mask_q,
                              gamma=args.mask_gamma, eps=args.mask_eps)

ALIGN = None
if args.alignment:
    _al = json.load(open(args.alignment))
    def _rod(rv):
        rv = np.asarray(rv, float); th = float(np.linalg.norm(rv)) + 1e-12
        k = rv/th
        K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
        return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)
    ALIGN = dict(s=float(_al['scale']),
                 R=torch.tensor(_rod(_al['rotvec']), dtype=torch.float32, device=DEVICE),
                 c=torch.tensor(_al['centre'],      dtype=torch.float32, device=DEVICE),
                 t=torch.tensor(_al['translation'], dtype=torch.float32, device=DEVICE))
    print(f"[ALIGN] s={ALIGN['s']:.4f}  t={_al['translation']}  rotvec={_al['rotvec']}", flush=True)


def maybe_align(v):
    if ALIGN is None:
        return v
    return ALIGN['s'] * ((v - ALIGN['c']) @ ALIGN['R'].T) + ALIGN['c'] + ALIGN['t']


# ── LoRA structures — rung5_colonly_lora.py L186-245, strict-loaded ──────────

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


def colonly_forward(dec_model, registry, slat):
    """rung5_colonly_lora.py L250-298, inference path."""
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
        frozen_h, lora_h = captured['frozen'], captured['lora']
        geom = frozen_h.feats[:, :COLOR_START].detach()
        col  = lora_h.feats[:, COLOR_START:COLOR_END].clamp(args.abs_min, args.abs_max)
        if W_VOX is not None:                       # rung11's forward, verbatim
            fcol = frozen_h.feats[:, COLOR_START:COLOR_END].detach()
            col  = fcol + (col - fcol) * W_VOX.to(col.dtype).unsqueeze(1)
        mixed = frozen_h.replace(torch.cat([geom, col], dim=1))
        meshes = dec_model.to_representation(mixed)
    return meshes[0]


def frozen_forward(dec_model, slat):
    with torch.no_grad():
        return dec_model(slat)[0]


# ── orbit camera — same construction as measure_geometry_change.py ───────────

def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    """theta=0 reproduces the confirmed front-view EXTRINSICS."""
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


def render(mesh, renderer, ext):
    mesh = filter_degenerate_faces(mesh)
    _sv = mesh.vertices
    mesh.vertices = maybe_align(mesh.vertices)       # rung13's render, verbatim
    res = renderer.render(mesh, ext,
                          INTRINSICS.to(DEVICE), return_types=['color', 'mask'])
    mesh.vertices = _sv
    m = res['mask']
    c = (res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))).detach().clamp(0, 1)
    return ((c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            int((m > 0.5).sum().item()))


LAB = 34
def strip(arrays, labels):
    H, W = arrays[0].shape[:2]
    pil = Image.fromarray(np.full((H + LAB, W * len(arrays), 3), 18, np.uint8))
    dr  = ImageDraw.Draw(pil)
    for i, (a, l) in enumerate(zip(arrays, labels)):
        pil.paste(Image.fromarray(a), (i * W, LAB))
        dr.rectangle([i * W, 0, (i + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
        dr.text((i * W + 10, 11), l, fill=(240, 240, 240))
    return pil


def main():
    assert CKPT.exists(),     f'missing {CKPT}'
    assert SLAT_NPZ.exists(), f'missing {SLAT_NPZ}'
    cfg = json.load(open(RUN_DIR / 'config.json'))
    ck  = torch.load(CKPT, map_location='cpu', weights_only=True)
    print(f'[CKPT] epoch={ck["epoch"]}  blocks={cfg["active_blocks"]}  '
          f'rank={cfg["rank"]}', flush=True)
    print(f'[SWEEP] {args.sweep}   angles={args.n_angles}   '
          f'elev={args.elevation}   radius={args.radius}', flush=True)

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
    print(f'[LORA] params={sum(p.numel() for p in registry.parameters()):,}  '
          f'(strict load OK)', flush=True)

    # GATE-cam: theta=0, elevation=0 must reproduce the confirmed front view.
    # If the orbit construction drifted, every angle in this video is wrong and
    # a "sticker" conclusion would be an artifact of the camera, not the adapter.
    _e0 = orbit_extrinsics(0.0, 0.0, 2.0).cpu()
    _d  = (_e0 - EXTRINSICS.cpu()).abs().max().item()
    print(f'[GATE-cam] max |orbit(0,0,2) - EXTRINSICS| = {_d:.3e}  (tol 1e-4)', flush=True)
    assert _d < 1e-4, f'GATE-cam FAILED: orbit camera does not match the front view ({_d:.3e})'
    print('[GATE-cam] PASSED', flush=True)

    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(parents=True, exist_ok=True)

    N = args.n_angles
    thetas = [360.0 * k / N for k in range(N)]
    if args.sweep == 'both':
        frames = [1 + int(round((args.n_frames - 1) * k / max(N - 1, 1))) for k in range(N)]
    else:
        frames = [args.frame] * N

    def slat_of(fi):
        return sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))

    mism, rows = 0, []
    cached_fi, m_f_cache, m_c_cache = None, None, None

    for k, (th, fi) in enumerate(zip(thetas, frames), start=1):
        ext = orbit_extrinsics(th, args.elevation, args.radius)

        # for --sweep angle the SLaT never changes, so decode once and reuse
        if fi != cached_fi:
            st = slat_of(fi)
            m_f_cache = frozen_forward(dec, st)
            m_c_cache = colonly_forward(dec, registry, st)
            cached_fi = fi
            del st
            gc.collect(); torch.cuda.empty_cache()

        nvf = int(m_f_cache.vertices.shape[0])
        nvc = int(m_c_cache.vertices.shape[0])
        if nvc != nvf:
            mism += 1

        gt = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
                      .resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        rgb_f, ar_f = render(m_f_cache, renderer, ext)
        rgb_c, ar_c = render(m_c_cache, renderer, ext)

        panels = [rgb_f, rgb_c]
        labels = [f'frozen TRELLIS  theta={th:5.1f}  V={nvf:,}',
                  f'{args.panel_name}  theta={th:5.1f}  V={nvc:,}'
                  f'  [{"SAME" if nvc == nvf else "DIFF"}]']
        if not args.no_gt:
            gt_lbl = (f'GT video  f{fi:04d}   (front view only)'
                      if abs(th) > 1e-6 else f'GT video  f{fi:04d}   theta=0')
            panels = [gt] + panels
            labels = [gt_lbl] + labels
        strip(panels, labels).save(fdir / f'orb_{k:04d}.png')

        rows.append(dict(k=k, theta=th, frame=fi, verts_frozen=nvf,
                         verts_colonly=nvc, area_frozen=ar_f, area_colonly=ar_c))
        if k % 15 == 0 or k == N:
            print(f'  {k}/{N}  theta={th:6.1f}  f{fi:03d}  '
                  f'frozen area={ar_f:,}  colonly area={ar_c:,}', flush=True)

    print(f'\n[GEOMETRY] vertex-count mismatches across the orbit: {mism}/{N}', flush=True)
    af = np.array([r['area_frozen'] for r in rows], float)
    ac = np.array([r['area_colonly'] for r in rows], float)
    print(f'[AREA] max |frozen-colonly| over all angles = {np.abs(af-ac).max():,.0f} px'
          f'   (0 means the silhouettes match at every viewpoint)', flush=True)

    ff = '/usr/bin/ffmpeg'
    pr = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    vid = OUT_DIR / (f'ORBIT_{args.sweep}_'
                 + ('frozen_vs_colonly' if args.no_gt else 'gt_frozen_colonly') + '.mp4')
    subprocess.run([ff, '-y', '-framerate', str(args.fps),
                    '-i', str(fdir / 'orb_%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)', flush=True)

    json.dump(dict(sweep=args.sweep, n_angles=N, elevation=args.elevation,
                   radius=args.radius, ckpt_epoch=ck['epoch'],
                   vertex_mismatches=mism, rows=rows),
              open(OUT_DIR / 'orbit.json', 'w'), indent=2)
    print(f'[DONE] {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
