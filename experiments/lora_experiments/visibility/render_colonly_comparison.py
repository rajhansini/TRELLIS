"""
render_colonly_comparison.py
----------------------------
GT video | frozen TRELLIS | rung5_colonly (two-pass splice), all 150 frames.

WHAT THIS IS CHECKING
  rung5_colonly_lora.py trains a block-level LoRA (all 12 decoder blocks, 4
  sublayers each) but keeps geometry from a FROZEN forward pass:

      geometry [0:53]   <- frozen pass, .detach()
      colour   [53:101] <- LoRA pass, .clamp(-9, 8)

  so the mesh should be bit-identical to the frozen decoder while the colour gets
  the upstream spatial reach that lets the lava pattern align with the video.

  Two things to look for, and the panel labels report both as numbers:

    1. VERTEX COUNT must equal the frozen panel's, every frame. If it does, the
       splice works and there is no second teapot. If it drifts, the splice is
       not doing what the source says and everything built on it is suspect.

    2. The HANDLE should be white. That run uses the UNION loss with the leaky
       `(gt < 0.99).any()` mask — the two bugs rung8 later fixed — so the
       mismatched rim is supervised toward white background. That artifact is
       expected here and is NOT geometry.

  The forward pass is copied from rung5_colonly_lora.py `colonly_forward`
  (L250-298); the LoRA structures are copied from its L186-245 and load with
  strict=True, so a structural mismatch raises rather than rendering a wrong
  adapter.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/render_colonly_comparison.py
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
ap.add_argument('--run',      default='rung5_colonly_r4_s6_5d6b1700')
ap.add_argument('--ckpt',     default='lora_best.pt')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--fps',      type=int, default=15)
ap.add_argument('--abs-min',  type=float, default=-9.0)
ap.add_argument('--abs-max',  type=float, default=8.0)
ap.add_argument('--alignment', default=None,
                help="alignment.json -> reproduce rung13's aligned render")
ap.add_argument('--panel-name', default='rung5_colonly (spliced geom)')
ap.add_argument('--out-dir',  default=None, type=Path)
args = ap.parse_args()

RUN_DIR  = _LORA / 'runs' / args.run
CKPT     = RUN_DIR / 'lora_ckpts' / args.ckpt
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
GT_DIR   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                '/outputs/teapot_lava_kling_premium'
                '/teapot_lava_kling_premium_front/all_frames_150')
OUT_DIR  = (args.out_dir or (RUN_DIR / 'comparison')).resolve()
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

# ── rung13: apply the solved mesh alignment to every render ──────────────────
ALIGN = None
if args.alignment:
    _al = json.load(open(args.alignment))
    def _rod(rv):
        rv = np.asarray(rv, float); th = float(np.linalg.norm(rv)) + 1e-12
        k = rv / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    ALIGN = dict(s=float(_al['scale']),
                 R=torch.tensor(_rod(_al['rotvec']), dtype=torch.float32, device=DEVICE),
                 c=torch.tensor(_al['centre'],      dtype=torch.float32, device=DEVICE),
                 t=torch.tensor(_al['translation'], dtype=torch.float32, device=DEVICE))
    print(f"[ALIGN] s={ALIGN['s']:.4f}  t={_al['translation']}", flush=True)


def maybe_align(v):
    if ALIGN is None:
        return v
    return ALIGN['s'] * ((v - ALIGN['c']) @ ALIGN['R'].T) + ALIGN['c'] + ALIGN['t']


# ── LoRA structures — rung5_colonly_lora.py L186-245 ─────────────────────────

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
    """
    rung5_colonly_lora.py L250-298, inference path (with_grad=False throughout).
    Pass 1 frozen -> geometry [0:53].  Pass 2 LoRA -> colour [53:101], clamped.
    """
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

        frozen_h = captured['frozen']
        lora_h   = captured['lora']
        frozen_geom = frozen_h.feats[:, :COLOR_START].detach()
        lora_col    = lora_h.feats[:, COLOR_START:COLOR_END].clamp(args.abs_min, args.abs_max)
        mixed_h     = frozen_h.replace(torch.cat([frozen_geom, lora_col], dim=1))
        meshes = dec_model.to_representation(mixed_h)
    return meshes[0]


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    mesh.faces = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > min_area]
    return mesh


def render(mesh, renderer):
    mesh = filter_degenerate_faces(mesh)
    _sv = mesh.vertices
    mesh.vertices = maybe_align(mesh.vertices)
    res = renderer.render(mesh,
                          EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
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
    print(f'[CKPT] {CKPT.name}  epoch={ck["epoch"]}  '
          f'blocks={cfg["active_blocks"]}  rank={cfg["rank"]}', flush=True)

    raw    = np.load(SLAT_NPZ)
    slats  = raw['slats']
    coords = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] {slats.shape}  coords {tuple(coords.shape)} (pinned)', flush=True)

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
    npar = sum(p.numel() for p in registry.parameters())
    print(f'[LORA] params={npar:,}  (strict load OK)', flush=True)

    renderer = make_renderer(DEVICE)
    fdir = OUT_DIR / 'frames'; fdir.mkdir(exist_ok=True)

    v_frz, v_co, a_frz, a_co, mismatches = [], [], [], [], 0

    for fi in range(1, args.n_frames + 1):
        gt = np.array(Image.open(GT_DIR / f'frame_{fi:04d}.png').convert('RGB')
                      .resize((RENDER_RES, RENDER_RES), Image.LANCZOS))
        st = sp.SparseTensor(
            feats=torch.from_numpy(slats[fi - 1].copy()).float().to(DEVICE),
            coords=coords.to(DEVICE))

        with torch.no_grad():
            m_f = dec(st)[0]
        nvf = int(m_f.vertices.shape[0])
        rgb_f, ar_f = render(m_f, renderer)

        m_c = colonly_forward(dec, registry, st)
        nvc = int(m_c.vertices.shape[0])
        rgb_c, ar_c = render(m_c, renderer)

        if nvc != nvf:
            mismatches += 1
        v_frz.append(nvf); v_co.append(nvc); a_frz.append(ar_f); a_co.append(ar_c)

        tag = 'SAME' if nvc == nvf else f'DIFF by {nvc-nvf:+,}'
        strip([gt, rgb_f, rgb_c],
              ['GT video',
               f'frozen TRELLIS  V={nvf:,}',
               f'{args.panel_name}  V={nvc:,}  [{tag}]']
              ).save(fdir / f'cmp_{fi:04d}.png')

        if fi % 25 == 0 or fi == args.n_frames:
            print(f'  {fi}/{args.n_frames}  frozen V={nvf:,} area={ar_f:,}   '
                  f'colonly V={nvc:,} area={ar_c:,}  {tag}', flush=True)
        del st, m_f, m_c
        gc.collect(); torch.cuda.empty_cache()

    print(f'\n[GEOMETRY CHECK] frames where vertex count differed from frozen: '
          f'{mismatches}/{args.n_frames}')
    print(f'  -> {"SPLICE CONFIRMED, geometry identical, no second teapot possible" if mismatches == 0 else "SPLICE IS NOT HOLDING — investigate before trusting this run"}',
          flush=True)
    af, ac = np.array(a_frz, float), np.array(a_co, float)
    print(f'[AREA] frozen mean {af.mean():,.0f}   colonly mean {ac.mean():,.0f}   '
          f'ratio {ac.mean()/af.mean():.4f}', flush=True)

    ff = '/usr/bin/ffmpeg'
    pr = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    vid = OUT_DIR / 'GT_vs_frozen_vs_result.mp4'
    subprocess.run([ff, '-y', '-framerate', str(args.fps),
                    '-i', str(fdir / 'cmp_%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)', flush=True)

    json.dump(dict(verts_frozen=v_frz, verts_colonly=v_co,
                   area_frozen=a_frz, area_colonly=a_co,
                   vertex_mismatches=mismatches, ckpt_epoch=ck['epoch']),
              open(OUT_DIR / 'comparison.json', 'w'), indent=2)
    print(f'[DONE] {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
