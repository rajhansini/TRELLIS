"""
test_alignment_compensation.py
------------------------------
STEP 1. Falsify (or confirm) the compensation story. No training.

THE CLAIM
  TRELLIS's mesh renders 11% too wide and ~16 px to the side of the video's
  teapot (measured: GT centroid x 265.1 / width 319, frozen 248.9 / width 355).
  The colour adapter cannot move the mesh — geometry is spliced from the frozen
  pass — so the only way it could reduce pixel error was to slide the TEXTURE
  across the surface until the lava landed on the right screen pixels.

THE PREDICTION THIS MAKES, AND THE WHOLE POINT OF THIS SCRIPT

    today   mesh -16px  +  texture +24px  ->  cancels     ->  NCC peak (0,  0)
    test    mesh ALIGNED +  texture +24px  ->  overshoots  ->  NCC peak (0,+24)

  Render the EXISTING trained adapter on the ALIGNED mesh. The compensation is
  now double-counted, so the render must get WORSE, in a specific direction.

  peak moves toward +24   -> mechanism confirmed, retraining is justified
  peak stays at (0, 0)    -> the story is WRONG. Say so and stop.

  This is a real falsification test: it can kill the theory for ~5 minutes of
  GPU instead of a 4 hour training run.

WHAT IS AND IS NOT TOUCHED
  Reads alignment.json (job 2148935) and rung5_colonly's lora_best.pt.
  Writes only into its own output directory. No training, no existing file
  modified, no checkpoint rewritten.

  The alignment transform was fitted on SILHOUETTE OVERLAP ONLY and never saw
  colour, so using it to diagnose a colour artifact is not circular.

OUTPUTS
  compensation.json   NCC peak offsets, the test result
  ncc_*.png           GT | colonly on UNALIGNED mesh | colonly on ALIGNED mesh
  orbit_*.png         the same three at several azimuths, so you can see whether
                      the texture is stuck to the surface or to the screen

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/test_alignment_compensation.py
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
ap.add_argument('--alignment', default=None, type=Path)
ap.add_argument('--frames',    default='75,120,150')
ap.add_argument('--thetas',    default='0,60,120,180,240,300')
ap.add_argument('--elevation', type=float, default=15.0)
ap.add_argument('--radius',    type=float, default=2.0)
ap.add_argument('--maxshift',  type=int, default=60)
ap.add_argument('--abs-min',   type=float, default=-9.0)
ap.add_argument('--abs-max',   type=float, default=8.0)
ap.add_argument('--out-dir',   default=None, type=Path)
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
ALIGN    = args.alignment or (_HERE / 'alignment' / 'alignment.json')
GT_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                '/outputs/teapot_lava_kling_premium'
                '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (_HERE / 'compensation_test')).resolve()
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


def rodrigues_np(rv):
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv) + 1e-12
    k  = rv / th
    K  = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


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


def render_rgb(mesh, renderer, ext, verts=None):
    saved = mesh.vertices
    if verts is not None:
        mesh.vertices = verts
    res = renderer.render(mesh, ext, INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
    mesh.vertices = saved
    m = res['mask']
    c = (res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))).detach().clamp(0, 1)
    return (c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def ncc_peak(a, b, maxshift):
    """Peak of the normalised cross-correlation of a against b."""
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / a.size
    cc = np.fft.fftshift(cc)
    cy, cx = np.array(cc.shape) // 2
    win = cc[cy - maxshift:cy + maxshift + 1, cx - maxshift:cx + maxshift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return int(iy - maxshift), int(ix - maxshift), float(win.max()), float(cc[cy, cx])


def lum_masked(rgb):
    L = rgb.astype(float).mean(axis=2)
    return np.where(rgb.min(axis=2) < 245, L, 0.0)


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
    assert ALIGN.exists(), f'missing alignment: {ALIGN}'
    al  = json.load(open(ALIGN))
    S   = float(al['scale'])
    RV  = np.array(al['rotvec'], dtype=np.float64)
    T   = np.array(al['translation'], dtype=np.float64)
    C   = np.array(al['centre'], dtype=np.float64)
    print('=' * 92)
    print('STEP 1 — does the adapter contain a compensating texture shift?')
    print(f'  alignment: s={S:.4f}  |rv|={np.linalg.norm(RV):.4f} rad  t={T.round(4).tolist()}')
    print(f'             IoU {al["iou_before"]:.4f} -> {al["iou_after"]:.4f}')
    print('  PREDICTION: colonly on the ALIGNED mesh should peak near dx=+16..+27,')
    print('              because the mesh no longer needs the compensation.')
    print('              If it stays at (0,0) the compensation story is WRONG.')
    print('=' * 92, flush=True)

    R_np = rodrigues_np(RV)
    R_t  = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    C_t  = torch.tensor(C,    dtype=torch.float32, device=DEVICE)
    T_t  = torch.tensor(T,    dtype=torch.float32, device=DEVICE)

    def align(v):
        return S * ((v - C_t) @ R_t.T) + C_t + T_t

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()
    cfg = json.load(open(RUN_DIR / 'config.json'))
    ck  = torch.load(CKPT, map_location='cpu', weights_only=True)

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
    print(f'[CKPT] epoch={ck["epoch"]}  params='
          f'{sum(p.numel() for p in registry.parameters()):,}  (strict load OK)', flush=True)

    renderer = make_renderer(DEVICE)
    ext0 = EXTRINSICS.to(DEVICE)
    frames = [int(x) for x in args.frames.split(',')]
    thetas = [float(x) for x in args.thetas.split(',')]

    rows = []
    for fi in frames:
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))
        mesh = filter_degenerate_faces(colonly_mesh(dec, registry, st))
        v    = mesh.vertices.detach().clone()
        va   = align(v)

        gt = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
                      .resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        r_un = render_rgb(mesh, renderer, ext0)
        r_al = render_rgb(mesh, renderer, ext0, va)

        g = lum_masked(gt)
        dy_u, dx_u, pk_u, z_u = ncc_peak(lum_masked(r_un), g, args.maxshift)
        dy_a, dx_a, pk_a, z_a = ncc_peak(lum_masked(r_al), g, args.maxshift)

        rows.append(dict(frame=fi,
                         unaligned=dict(dy=dy_u, dx=dx_u, ncc=pk_u, ncc_zero=z_u),
                         aligned=dict(dy=dy_a, dx=dx_a, ncc=pk_a, ncc_zero=z_a)))
        print(f'\n[frame {fi}]  colonly texture, training camera')
        print(f'   on UNALIGNED mesh : peak (dy={dy_u:+3d}, dx={dx_u:+3d})  '
              f'ncc={pk_u:.4f}  ncc@0={z_u:.4f}')
        print(f'   on ALIGNED   mesh : peak (dy={dy_a:+3d}, dx={dx_a:+3d})  '
              f'ncc={pk_a:.4f}  ncc@0={z_a:.4f}', flush=True)

        strip([gt, r_un, r_al],
              [f'GT  f{fi:04d}',
               f'colonly on UNALIGNED mesh   NCC peak dx={dx_u:+d}',
               f'colonly on ALIGNED mesh     NCC peak dx={dx_a:+d}']
              ).save(OUT_DIR / f'ncc_f{fi:04d}.png')

        if fi == frames[0]:
            for th in thetas:
                e = orbit_extrinsics(th, args.elevation, args.radius)
                strip([render_rgb(mesh, renderer, e),
                       render_rgb(mesh, renderer, e, va)],
                      [f'colonly UNALIGNED  theta={th:.0f}',
                       f'colonly ALIGNED    theta={th:.0f}']
                      ).save(OUT_DIR / f'orbit_f{fi:04d}_t{int(th):03d}.png')

        del st, mesh, v, va
        gc.collect(); torch.cuda.empty_cache()

    dxu = np.array([r['unaligned']['dx'] for r in rows], float)
    dxa = np.array([r['aligned']['dx'] for r in rows], float)
    print('\n' + '=' * 92)
    print(f'{"frame":>7}{"dx unaligned":>16}{"dx aligned":>14}{"shift":>10}')
    print('-' * 92)
    for r in rows:
        d = r['aligned']['dx'] - r['unaligned']['dx']
        print(f'{r["frame"]:>7}{r["unaligned"]["dx"]:>16d}{r["aligned"]["dx"]:>14d}{d:>+10d}')
    print('-' * 92)
    moved = float(np.mean(dxa - dxu))
    print(f'  mean dx: unaligned {dxu.mean():+.1f}  ->  aligned {dxa.mean():+.1f}   '
          f'(moved {moved:+.1f} px)')
    if abs(moved) >= 8:
        verdict = ('CONFIRMED — aligning the mesh exposes a baked-in texture shift. '
                   'The adapter was compensating. Retraining on the aligned mesh is '
                   'justified.')
    elif abs(moved) <= 3:
        verdict = ('REFUTED — the texture did not move when the mesh did. The '
                   'compensation story is WRONG and retraining on the aligned mesh '
                   'is not justified by this evidence.')
    else:
        verdict = (f'INCONCLUSIVE — moved {moved:+.1f} px, between the +/-3 and +/-8 '
                   f'thresholds set in advance. Do not build on this.')
    print(f'\n  VERDICT: {verdict}')
    print('=' * 92, flush=True)

    json.dump({'alignment': {'scale': S, 'rotvec': RV.tolist(),
                             'translation': T.tolist()},
               'ckpt_epoch': ck['epoch'], 'rows': rows,
               'mean_dx_unaligned': float(dxu.mean()),
               'mean_dx_aligned': float(dxa.mean()),
               'moved_px': moved, 'verdict': verdict},
              open(OUT_DIR / 'compensation.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
