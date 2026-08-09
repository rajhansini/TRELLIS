"""
rung9_geomcolor_lora.py
-----------------------
rung8, with the LoRA write window widened from colour only to GEOMETRY + COLOUR.

ONE VARIABLE vs rung8. Loss, mask, normaliser, coords, rank, LR, epochs, seed,
LOSS_SCALE, held-out split, GT threshold — all identical. The only difference is
which out_layer channels receive the adapter delta.

WHY
  rung8 wrote [53:101] (colour). Vertex positions are computed by FlexiCubes from
  sdf [0:8] and deform [8:32] only, so rung8 could not move the mesh however hard
  it trained. Measured tracking ratio of the frozen decoder was 0.063 — TRELLIS's
  silhouette spread is 0.30% where the GT video's is 5.43%. rung9 tests whether
  opening sdf+deform to the adapter closes that gap.

  Pinning `coords` does NOT freeze geometry. `coords` fixes the 7,301-voxel
  lattice; the surface moves continuously inside each cube via sdf and deform.
  Coords stay pinned here on purpose: with per-frame coords the mesh already
  jumps on its own (measured 19.9x the pinned flicker), and any silhouette
  movement rung9 produced would be unattributable.

CHANNEL LAYOUT  (trellis/representations/mesh/cube2mesh.py, LAYOUTS)
  [0:8]     sdf                   8 cube corners
  [8:32]    deform                8 corners x 3 xyz          -> vertex POSITION
  [32:53]   flexicubes weights    alpha 8 + beta 12 + gamma 1 -> topology
  [53:101]  colour                8 corners x 6
                                    = 24 albedo + 24 shading normal

  --write-blocks color        [53:101]                 == rung8, control arm
  --write-blocks geom+color   [0:32] and [53:101]      == DEFAULT, this run
  --write-blocks all101       [0:101]                  includes [32:53]

  [32:53] is excluded by default. Those weights steer dual-vertex placement and
  the sign-based topology decision; perturbing them changes WHICH triangles
  exist rather than where the surface is, and that is the failure mode that
  shreds FlexiCubes meshes. all101 exists so the question can be answered by
  ablation rather than assertion.

CLAMP
  ABS_MIN/ABS_MAX were fitted to the colour block in v4 and are applied ONLY to
  [53:101] here. Applying a colour-fitted bound to sdf would be an unchosen
  geometry constraint. rung8 logged clip_frac 0.0000 for all 30 epochs, so the
  clamp has never once fired; it is kept solely so the colour half stays a true
  one-variable comparison against rung8.

WHERE THE SHAPE GRADIENT COMES FROM
  The renderer calls dr.antialias on both colour and mask, so vertex positions
  are differentiable, silhouette included. But the loss is masked to
  live_render_mask & gt_mask, which excludes the rim where the two silhouettes
  DISAGREE — exactly where the largest shape error lives. So rung9 receives:
    - full gradient from interior shading as the surface tilts
    - antialiased boundary gradient only along the AGREEING boundary
    - no gradient at all from the disagreeing rim
  A partial result (ratio improves but does not reach 1.0) is therefore the
  expected outcome, and points at a silhouette term as the next rung, not at
  geometry channels being the wrong idea.

SUCCESS CRITERION, stated in advance
  tracking_ratio, logged every epoch on 10 frames, definition identical to
  measure_geometry_change.py:
      ratio = (rendered silhouette area spread %) / (GT silhouette area spread %)
      spread % = (max - min) / mean * 100
  frozen decoder baseline = 0.063. Anything above 0.20 is a real response.

  GUARD: dB (blue channel shift vs frozen) must stay near rung8's +0.105 and
  must not climb back toward v4's +0.238. If tracking improves while dB worsens,
  the adapter is deforming the mesh to fake brightness rather than tracking
  shape, and the run is a failure even with a good ratio.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/rung9_geomcolor_lora.py \
    --rank 4 --epochs 30 --seed 6 --write-blocks geom+color \
    2>&1 | tee experiments/lora_experiments/visibility/logs/rung9.log
"""

import sys, os, argparse as _ap, hashlib, json
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE  = Path(__file__).resolve().parent
_LORA  = _HERE.parent
_ROOT  = _LORA.parent.parent
_PIPE  = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_RUNS  = _HERE / 'runs'
_RUNS.mkdir(parents=True, exist_ok=True)

_ap_ = _ap.ArgumentParser()
_ap_.add_argument('--rank',       type=int,   default=4)
_ap_.add_argument('--seed',       type=int,   default=6)
_ap_.add_argument('--epochs',     type=int,   default=30)
_ap_.add_argument('--abs-min',    type=float, default=-9.0)
_ap_.add_argument('--abs-max',    type=float, default=8.0)
_ap_.add_argument('--lambda-reg', type=float, default=0.0,
                  help='0.0 matches rung8. >0 enables the FIXED regulariser '
                       '(applied to every written block).')
_ap_.add_argument('--write-blocks', default='geom+color',
                  choices=['color', 'geom+color', 'all101'],
                  help="which out_layer channels the LoRA writes. 'color' "
                       "reproduces rung8 exactly.")
_ap_.add_argument('--smoke',      action='store_true')
_ap_.add_argument('--diag-every', type=int,   default=5)
_ap_.add_argument('--loss-norm',  default='v4compat',
                  choices=['v4compat', 'intersection'],
                  help="v4compat keeps rung8's gradient scale (true one-variable "
                       "ablation). intersection is a proper mean, ~9.5x.")
_ap_.add_argument('--v4-run-id',  default='c85c888f',
                  help='run whose slat_cache.npz is reused (pinned frame-75 coords)')
args = _ap_.parse_args()

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED     = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES       = 150
STRUCT_SEED    = 42
STEPS          = 25
RESCALE_T      = 3.0
LR             = 1e-4
LOSS_SCALE0    = 4096.0
GRAD_CLIP      = 1.0
W_LPIPS        = 0.1
DEC_DIM        = 768
DEC_OUT_IN_DIM = DEC_DIM // 8   # 96

# out_layer channel layout, from cube2mesh.py LAYOUTS
SDF_START,    SDF_END    = 0,  8
DEFORM_START, DEFORM_END = 8,  32
WEIGHT_START, WEIGHT_END = 32, 53
COLOR_START,  COLOR_END  = 53, 101
COLOR_DIM = COLOR_END - COLOR_START     # 48
GEOM_DIM  = DEFORM_END - SDF_START      # 32

_BLOCK_SETS = {
    'color':      [(COLOR_START, COLOR_END)],
    'geom+color': [(SDF_START, DEFORM_END), (COLOR_START, COLOR_END)],
    'all101':     [(0, 101)],
}
BLOCKS  = _BLOCK_SETS[args.write_blocks]
OUT_DIM = sum(e - s for s, e in BLOCKS)

# byte offset of each block inside the LoRA output vector
_OFFS, _o = [], 0
for _s, _e in BLOCKS:
    _OFFS.append(_o); _o += _e - _s
assert _o == OUT_DIM

ABS_MIN    = args.abs_min
ABS_MAX    = args.abs_max
LAMBDA_REG = args.lambda_reg

HELD_OUT = list(range(5, 151, 10))
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

_CFG = dict(
    variant='geom_color_outlayer_intersection',
    write_blocks=args.write_blocks,
    loss_region='intersection(live_render_mask & gt_mask)',
    rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    abs_min=ABS_MIN, abs_max=ABS_MAX, lambda_reg=LAMBDA_REG,
    gt_bg_thresh=0.95, loss_norm=args.loss_norm, held_out=HELD_OUT,
    coords='pinned_frame75', struct_seed=STRUCT_SEED,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = f'rung9_{args.write_blocks.replace("+", "-")}_r{args.rank}_s{args.seed}_{RUN_ID}'
_OUT   = _RUNS / _LABEL
_CKPT  = _OUT / 'lora_ckpts'
_DIAG  = _OUT / 'diag_renders'
_LOGD  = _OUT / 'logs'

for _d in (_OUT, _CKPT, _DIAG, _LOGD):
    _d.mkdir(parents=True, exist_ok=True)


class _Tee:
    """stdout -> console + train.log (line buffered, survives requeue via 'a')."""
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(_OUT / 'train.log')
sys.stderr = sys.stdout


def _ensure_nvdiffrast():
    import subprocess, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm{p.major}{p.minor}'
    arch_str = f'{p.major}.{p.minor}'
    local    = f'/tmp/nvdiffrast_{arch_tag}'
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
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')
    print(f'[NVDIFF] building for {arch_tag} -> {local}', flush=True)
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    print('[NVDIFF] build done — restarting', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import math, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as _ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM  = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]
GEOM_FRAMES = list(range(1, N_FRAMES + 1, 15))          # 10 frames
ORBIT_ANGLES = [0.0, 90.0, 180.0, 270.0]


class OutLayerLoRA(nn.Module):
    """
    Same shape as rung8's adapter except for B's row count.

        delta = B @ (A @ x_v)        A: rank x 96      B: OUT_DIM x rank

    OUT_DIM is 48 for --write-blocks color (identical to rung8, checkpoints
    interchangeable), 80 for geom+color, 101 for all101. B is zero-init so
    epoch 0 is bit-identical to the frozen decoder in every configuration.
    """
    def __init__(self, rank: int, in_dim: int = DEC_OUT_IN_DIM,
                 out_dim: int = OUT_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


_DTYPE_REPORTED = False


def _note_dtype(x_dt, d_dt, o_dt):
    """Print the hook's working dtype once. fp16 sets the delta resolution floor."""
    global _DTYPE_REPORTED
    _DTYPE_REPORTED = True
    eps = float(torch.finfo(o_dt).eps) if o_dt.is_floating_point else float('nan')
    print(f'[DTYPE] out_layer in={x_dt}  lora_delta={d_dt}  out={o_dt}   '
          f'eps={eps:.2e}  -> deltas below ~{eps:.0e} relative to a channel of '
          f'magnitude 1 are erased by the forward pass', flush=True)


def _clamp_colour_only(rw, s, e):
    """
    Apply the colour clamp to the [53:101] portion of a block and nothing else.

    Block (53,101)  -> clamp the whole thing (rung8 behaviour)
    Block (0,32)    -> untouched; a colour-fitted bound is not a geometry prior
    Block (0,101)   -> clamp columns 53..101 only
    """
    lo, hi = max(s, COLOR_START), min(e, COLOR_END)
    if lo >= hi:
        return rw
    if lo == s and hi == e:
        return rw.clamp(ABS_MIN, ABS_MAX)
    out = rw.clone()
    out[:, lo - s:hi - s] = rw[:, lo - s:hi - s].clamp(ABS_MIN, ABS_MAX)
    return out


def geomcol_forward(dec_model, lora, slat_norm, with_grad=True, return_feats=False):
    """
    rung8's colonly_forward generalised to N blocks.

    The .clone() on frozen is load-bearing for the same reason as in rung8:
    new_feats[:, s:e] is a VIEW, and the in-place write below would otherwise
    overwrite the captured frozen values, making the regulariser evaluate
    (raw - clamp(raw)) ~ 0. That was the v4 bug that made lambda_reg a no-op.
    """
    captured = {}

    def _hook(mod, inp, out):
        x_feats   = inp[0].feats
        delta     = lora(x_feats)                 # (N_fine, OUT_DIM)
        new_feats = out.feats.clone()
        assert new_feats.shape[1] == 101, (
            f'out_layer emits {new_feats.shape[1]} channels, not 101 — the '
            f'block offsets in this file are wrong for this checkpoint')
        if not _DTYPE_REPORTED:
            # fp16 here means 1.0 + 1e-4 rounds back to 1.0, so sdf deltas below
            # ~1e-3 are erased by the forward pass regardless of the gradient.
            # See GATE-geom for the measured floor.
            _note_dtype(x_feats.dtype, delta.dtype, out.feats.dtype)
        frozen_l, raw_l = [], []
        for (s, e), off in zip(BLOCKS, _OFFS):
            w      = e - s
            frozen = new_feats[:, s:e].clone()
            raw    = frozen + delta[:, off:off + w]
            new_feats[:, s:e] = _clamp_colour_only(raw, s, e)
            frozen_l.append(frozen.detach()); raw_l.append(raw)
        captured['frozen'] = torch.cat(frozen_l, dim=1)
        captured['raw']    = torch.cat(raw_l, dim=1)
        return out.replace(new_feats)

    _ctx = torch.no_grad() if not with_grad else _nullctx()
    with _ctx:
        h = dec_model.out_layer.register_forward_hook(_hook)
        meshes = dec_model(slat_norm)
        h.remove()

    mesh = meshes[0]
    if return_feats:
        return mesh, captured['raw'], captured['frozen']
    return mesh


def encode_frame(dino_model, frame_idx):
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


GT_BG_THRESH = 0.95


def load_gt(frame_idx):
    """Tight GT teapot mask. min(RGB) < 0.95; see rung8 for why 0.99-any is leaky."""
    img    = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img    = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt     = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = gt.min(dim=0).values < GT_BG_THRESH
    return gt, gt_mask


def load_gt_v4mask(frame_idx):
    """v4's exact (leaky) mask — used ONLY for reporting v4-comparable metrics."""
    img    = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img    = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt     = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    return gt, (gt < 0.99).any(dim=0)


def load_render_mask():
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)


def masked_psnr(pred, gt, mask):
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred, gt):
    p = pred.permute(1, 2, 0).cpu().numpy()
    g = gt.permute(1, 2, 0).cpu().numpy()
    return float(_ssim(p, g, data_range=1.0, channel_axis=2))


def intersection_loss(rendered, gt, live_rmask, gt_mask, lpips_fn, norm_px=None):
    """Byte-identical to rung8. Do not change — this is the controlled variable."""
    m = (live_rmask & gt_mask).float()
    n = m.sum()
    if n < 1.0:
        z = rendered.sum() * 0.0
        return z, z, z, 0.0
    denom = (norm_px if norm_px is not None else n)
    mse = ((rendered - gt) ** 2 * m).sum() / (denom * 3 + 1e-8)
    r   = rendered * m + (1 - m)
    g   = gt       * m + (1 - m)
    lp  = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).mean()
    return mse, lp, mse + W_LPIPS * lp, float(n)


def union_metrics(rendered, gt, static_rmask, gt_mask, lpips_fn):
    """v4's exact metric, kept unchanged so the numbers stay comparable."""
    m = (static_rmask | gt_mask).float()
    mse = ((rendered - gt) ** 2 * m).sum() / (m.sum() * 3 + 1e-8)
    r   = rendered * m + (1 - m)
    g   = gt       * m + (1 - m)
    with torch.no_grad():
        lp = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).item()
    return mse.item(), lp


def masked_reg(raw, frozen):
    return (raw - frozen).pow(2).mean()


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def render_mesh(mesh, renderer, ext=None):
    """Returns (composited colour, LIVE boolean silhouette)."""
    ext  = EXTRINSICS.to(DEVICE) if ext is None else ext
    intr = INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    m    = res['mask']
    colour = res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))
    return colour, (m > 0.5)


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


def chamfer(a, b, chunk=2048):
    def one_way(x, y):
        tot, n = 0.0, 0
        for i in range(0, x.shape[0], chunk):
            d = torch.cdist(x[i:i + chunk], y)
            tot += d.min(dim=1).values.sum().item()
            n   += d.shape[0]
        return tot / max(n, 1)
    return 0.5 * (one_way(a, b) + one_way(b, a))


def subsample(v, n, gen):
    if v.shape[0] <= n:
        return v
    idx = torch.randperm(v.shape[0], generator=gen, device='cpu')[:n]
    return v[idx.to(v.device)]


def iou(a, b):
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


def _slat_from_cache(slat_cache, fi):
    feats, coords = slat_cache[fi]
    return sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))


def make_strip(panels, cell=320, label_h=28):
    canvas = Image.new('RGB', (cell * len(panels), cell + label_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (img_src, lbl) in enumerate(panels):
        if isinstance(img_src, (str, Path)):
            img = Image.open(img_src).convert('RGB').resize((cell, cell), Image.LANCZOS)
        else:
            arr = (img_src.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((cell, cell), Image.LANCZOS)
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col * cell, 0, (col + 1) * cell - 1, label_h - 1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0, 0), lbl)[2]
        except Exception: tw = len(lbl) * 7
        draw.text((col * cell + (cell - tw) // 2, 6), lbl, fill=(210, 210, 210))
    return canvas


def measure_channel_shift(dec_model, lora, slat_cache, renderer, frames):
    """
    THE COLOUR GUARD. mean(lora - frozen) per RGB channel over the rendered object.
      v4    reached [+0.2457, +0.2233, +0.2380]   (white cast)
      rung8 reached [ ...   ,  +0.086 ,  +0.105]  (fixed)
    rung9 must not regress toward v4.
    """
    dec_model.eval()
    d = torch.zeros(3, device=DEVICE); n = 0
    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = geomcol_forward(dec_model, lora, slat, with_grad=False)
            c_l, m_l = render_mesh(mesh, renderer)
            meshes_b = dec_model(slat)
            c_f, m_f = render_mesh(meshes_b[0], renderer)
            obj = m_f
            if obj.sum() == 0:
                continue
            d += (c_l.clamp(0, 1) - c_f.clamp(0, 1))[:, obj].sum(dim=1)
            n += int(obj.sum())
            del slat, mesh, c_l, c_f, meshes_b
            gc.collect(); torch.cuda.empty_cache()
    return (d / max(n, 1)).cpu().numpy()


_GT_AREA_CACHE = {}


def _gt_areas(frames):
    for fi in frames:
        if fi not in _GT_AREA_CACHE:
            _, gm = load_gt(fi)
            _GT_AREA_CACHE[fi] = int(gm.sum().item())
    return np.array([_GT_AREA_CACHE[fi] for fi in frames], dtype=np.float64)


def _spread_pct(a):
    return float((a.max() - a.min()) / a.mean() * 100.0) if a.mean() > 0 else 0.0


def measure_geometry_tracking(dec_model, lora, slat_cache, renderer, frames):
    """
    THE SUCCESS CRITERION.

    Definition is identical to measure_geometry_change.py so the number is
    directly comparable to the frozen-decoder baseline of 0.063:

        spread %      = (max - min) / mean * 100     over the frame set
        tracking_ratio = spread%(rendered silhouette) / spread%(GT silhouette)

    Front view (0 deg) only, which is the view the loss supervises.
    """
    dec_model.eval()
    areas, verts, ious = [], [], []
    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = geomcol_forward(dec_model, lora, slat, with_grad=False)
            nv   = int(mesh.vertices.shape[0])
            _, rm = render_mesh(mesh, renderer)
            _, gm = load_gt(fi)
            areas.append(int(rm.sum().item())); verts.append(nv)
            ious.append(iou(rm, gm))
            del slat, mesh, rm, gm
            gc.collect(); torch.cuda.empty_cache()
    a_tr = np.array(areas, dtype=np.float64)
    a_gt = _gt_areas(frames)
    s_tr, s_gt = _spread_pct(a_tr), _spread_pct(a_gt)
    return {
        'spread_render_pct': s_tr,
        'spread_gt_pct':     s_gt,
        'tracking_ratio':    (s_tr / s_gt) if s_gt > 1e-9 else float('nan'),
        'verts_mean':        float(np.mean(verts)),
        'verts_spread_pct':  _spread_pct(np.array(verts, dtype=np.float64)),
        'iou_gt_mean':       float(np.mean(ious)),
        'areas':             areas,
    }


def measure_regions(dec_model, lora, slat_cache, renderer, frames):
    """Log |A|, |B| and silhouette IoU so the loss-region premise is auditable."""
    dec_model.eval()
    tA = tB = tU = 0
    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = geomcol_forward(dec_model, lora, slat, with_grad=False)
            _, rm = render_mesh(mesh, renderer)
            _, gm = load_gt(fi)
            tA += (rm & gm).sum().item()
            tB += (rm & ~gm).sum().item()
            tU += (rm | gm).sum().item()
            del slat, mesh, rm, gm
            gc.collect(); torch.cuda.empty_cache()
    return tA, tB, (tA / max(tU, 1))


def evaluate_frames(dec_model, lora, slat_cache, renderer, static_rmask,
                    lpips_fn, frame_list):
    """v4-identical metrics (union region, static mask) + intersection PSNR."""
    dec_model.eval()
    per_frame = []
    for fi in frame_list:
        slat_norm = _slat_from_cache(slat_cache, fi)
        mesh  = geomcol_forward(dec_model, lora, slat_norm, with_grad=False)
        colour, live_rm = render_mesh(mesh, renderer)
        render = colour.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)
        _,  gm_v4   = load_gt_v4mask(fi)
        psnr_u = masked_psnr(render, gt, static_rmask)
        inter  = (live_rm & gt_mask)
        psnr_i = masked_psnr(render, gt, inter) if inter.sum() > 0 else float('nan')
        ssim   = compute_ssim(render, gt)
        _, lp  = union_metrics(render, gt, static_rmask, gm_v4, lpips_fn)
        per_frame.append({'frame': fi, 'psnr': psnr_u, 'psnr_inter': psnr_i,
                          'ssim': ssim, 'lpips': lp,
                          'iou': iou(live_rm, gt_mask)})
        del render, gt, gt_mask, gm_v4, mesh, colour, live_rm
        gc.collect(); torch.cuda.empty_cache()
    f = lambda k: [r[k] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(f('psnr'))), 'psnr_std': float(np.std(f('psnr'))),
        'psnr_inter_mean': float(np.nanmean(f('psnr_inter'))),
        'ssim_mean': float(np.mean(f('ssim'))), 'ssim_std': float(np.std(f('ssim'))),
        'lpips_mean': float(np.mean(f('lpips'))), 'lpips_std': float(np.std(f('lpips'))),
        'iou_mean': float(np.mean(f('iou'))),
        'per_frame': per_frame,
    }


def final_geometry_report(dec_model, lora, slat_cache, renderer, out_json):
    """
    Full geometry diagnostic, same shape as measure_geometry_change.py so the
    result drops straight into the comparison against the 0.063 baseline.
    Four orbit views + chamfer against frame 1 + the front-view tracking ratio.
    """
    dec_model.eval()
    frames = list(range(1, N_FRAMES + 1, 10))
    if frames[-1] != N_FRAMES:
        frames.append(N_FRAMES)
    exts = {th: orbit_extrinsics(th, 0.0, 2.0) for th in ORBIT_ANGLES}
    gen  = torch.Generator().manual_seed(0)

    ref_pts, ref_masks, ref_gt = None, {}, None
    rows, noise_floor = [], 0.0

    print(f'\n{"frame":>6} {"verts":>9} {"faces":>9} {"chamfer_f1":>12} '
          f'{"sil_0deg":>10} {"IoU_f1":>8} {"GT_area":>9} {"GT_IoU":>8}', flush=True)

    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = geomcol_forward(dec_model, lora, slat, with_grad=False)
            nv, nf = int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])
            masks = {}
            for th in ORBIT_ANGLES:
                _, m = render_mesh(mesh, renderer, exts[th])
                masks[th] = m
            _, gm = load_gt(fi)
            pts = subsample(mesh.vertices.detach(), 20000, gen)
            if ref_pts is None:
                ref_pts, ref_masks, ref_gt = pts.clone(), {k: v.clone() for k, v in masks.items()}, gm.clone()
                noise_floor = chamfer(ref_pts, ref_pts.clone())
                print(f'[NOISE] chamfer(f1,f1) = {noise_floor:.3e}', flush=True)
            cd = chamfer(pts, ref_pts) if fi != frames[0] else 0.0
            rows.append({
                'frame': fi, 'verts': nv, 'faces': nf, 'chamfer_vs_f1': float(cd),
                'sil_area': {str(th): int(masks[th].sum().item()) for th in ORBIT_ANGLES},
                'sil_iou_f1': {str(th): float(iou(masks[th], ref_masks[th])) for th in ORBIT_ANGLES},
                'gt_area': int(gm.sum().item()), 'gt_iou_f1': float(iou(gm, ref_gt)),
            })
            print(f'{fi:>6} {nv:>9,} {nf:>9,} {cd:>12.6f} '
                  f'{rows[-1]["sil_area"]["0.0"]:>10,} '
                  f'{rows[-1]["sil_iou_f1"]["0.0"]:>8.5f} '
                  f'{rows[-1]["gt_area"]:>9,} {rows[-1]["gt_iou_f1"]:>8.5f}', flush=True)
            del slat, mesh, masks, gm, pts
            gc.collect(); torch.cuda.empty_cache()

    a_tr = np.array([r['sil_area']['0.0'] for r in rows], dtype=np.float64)
    a_gt = np.array([r['gt_area'] for r in rows], dtype=np.float64)
    rng_tr, rng_gt = _spread_pct(a_tr), _spread_pct(a_gt)
    ratio  = (rng_tr / rng_gt) if rng_gt > 1e-9 else float('nan')
    cd_max = max(r['chamfer_vs_f1'] for r in rows)
    # chamfer(X, X) is exactly 0 — every point is its own nearest neighbour — so
    # the "noise floor" is a self-consistency check (must be 0), not a scale.
    snr = float('inf') if noise_floor <= 1e-12 else cd_max / noise_floor

    print(f'\n  rendered silhouette spread : {rng_tr:.2f}% of mean')
    print(f'  GT silhouette spread       : {rng_gt:.2f}% of mean')
    print(f'  max chamfer to frame 1     : {cd_max:.6f}'
          + ('   (self-chamfer 0.0 as expected)' if noise_floor <= 1e-12
             else f'   (noise floor {noise_floor:.3e}, SNR {snr:.1f}x)'))
    print(f'\n  TRACKING RATIO = {ratio:.3f}    (frozen decoder baseline: 0.063)')
    if   ratio > 0.6: verdict = 'STRONG tracking — geometry LoRA is doing the work.'
    elif ratio > 0.2: verdict = 'PARTIAL tracking — real response, silhouette term is the next rung.'
    else:             verdict = 'NO tracking — opening sdf/deform was not sufficient.'
    print(f'  -> {verdict}', flush=True)

    json.dump({'rows': rows, 'render_spread_pct': rng_tr, 'gt_spread_pct': rng_gt,
               'tracking_ratio': ratio, 'max_chamfer': cd_max,
               'chamfer_noise_floor': noise_floor, 'chamfer_snr': snr,
               'baseline_frozen_ratio': 0.063, 'verdict': verdict},
              open(out_json, 'w'), indent=2)
    return ratio, verdict


def save_curves(history):
    if len(history) < 2:
        return
    ep  = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    fig.suptitle(f'Rung 9 geometry+colour LoRA — {_LABEL}', fontsize=11)
    axes[0].plot(ep, [r['loss_mse'] for r in history], 'o-', color='#e06c75', label='MSE')
    axes[0].plot(ep, [r['loss_lpips'] for r in history], 's-', color='#d19a66', label='LPIPS')
    axes[0].plot(ep, [r['loss_total'] for r in history], '^-', color='#c678dd', label='total')
    axes[0].set_title('Train loss (intersection)'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, [r['held_psnr'] for r in history], 'o-', color='#98c379', label='union (v4-comparable)')
    axes[1].plot(ep, [r['held_psnr_inter'] for r in history], 's--', color='#56b6c2', label='intersection')
    axes[1].set_title('Held-out PSNR (dB)'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, [r['B_norm'] for r in history], 'o-', color='#61afef', label='||B|| all')
    axes[2].plot(ep, [r['B_norm_geom'] for r in history], 's--', color='#e5c07b', label='||B|| geom rows')
    axes[2].plot(ep, [r['B_norm_col'] for r in history], '^--', color='#c678dd', label='||B|| colour rows')
    axes[2].axhline(1.6487, color='#e06c75', ls='--', lw=1, label='v4 final 1.65')
    axes[2].set_title('||B||'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)
    for i, (c, col) in enumerate(zip('RGB', ('#e06c75', '#98c379', '#61afef'))):
        axes[3].plot(ep, [r['shift'][i] for r in history], 'o-', color=col, label=f'd{c}')
    for y, lbl, st in ((0.238, 'v4 dB +0.238', '--'), (0.105, 'rung8 dB +0.105', ':')):
        axes[3].axhline(y, color='#888', ls=st, lw=1, label=lbl)
    axes[3].set_title('Colour guard: shift (lora - frozen)'); axes[3].legend(fontsize=7)
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(ep, [r['tracking_ratio'] for r in history], 'o-', color='#98c379')
    # measured on GEOM_FRAMES with B=0, not the 0.063 from a different frame set
    _br = history[0].get('base_ratio', float('nan'))
    axes[4].axhline(_br,  color='#e06c75', ls='--', lw=1,
                    label=f'frozen, same frames: {_br:.3f}')
    axes[4].axhline(0.20, color='#888',    ls=':',  lw=1, label='response threshold 0.20')
    axes[4].set_title('SUCCESS CRITERION: tracking ratio'); axes[4].legend(fontsize=7)
    axes[4].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


def find_latest_ckpt():
    c = sorted(_CKPT.glob('lora_e*.pt'))
    return c[-1] if c else None


def _b_row_norms(lora):
    """
    ||B|| split by which block each row writes, so a dead half is visible.

    Diagnostic only. Under --write-blocks all101 the 'geom' bucket is [0:53],
    i.e. it also contains the FlexiCubes weight rows [32:53]; under the default
    geom+color it is exactly [0:32].
    """
    B = lora.B.detach().float()
    tot = B.norm().item()
    g = c = 0.0
    for (s, e), off in zip(BLOCKS, _OFFS):
        w   = e - s
        sub = B[off:off + w]
        lo, hi = max(s, COLOR_START), min(e, COLOR_END)
        if lo >= hi:
            g += sub.norm().item() ** 2
        elif lo == s and hi == e:
            c += sub.norm().item() ** 2
        else:
            c += sub[lo - s:hi - s].norm().item() ** 2
            g += (sub.norm().item() ** 2 - sub[lo - s:hi - s].norm().item() ** 2)
    return tot, math.sqrt(max(g, 0.0)), math.sqrt(max(c, 0.0))


def main():
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    print('=' * 78)
    print('Rung 9 — GEOMETRY + COLOUR LoRA (out_layer hook, pinned coords)')
    print(f'  write blocks  : {args.write_blocks}  ->  {BLOCKS}   OUT_DIM={OUT_DIM}')
    print(f'  loss region   : live_render_mask & gt_mask   (unchanged from rung8)')
    print(f'  coords        : PINNED to frame 75, struct seed {STRUCT_SEED}')
    print(f'  rank          : {args.rank}     seed: {args.seed}     epochs: {EPOCHS}')
    print(f'  lambda_reg    : {LAMBDA_REG}')
    print(f'  colour clamp  : [{ABS_MIN}, {ABS_MAX}] on [53:101] ONLY   '
          f'LOSS_SCALE: {LOSS_SCALE0:.0f}')
    print(f'  run_id        : {RUN_ID}')
    print(f'  output        : {_OUT}')
    print('  SUCCESS  : tracking_ratio > 0.20   (frozen baseline 0.063)')
    print('  GUARD    : dB must stay near rung8 +0.105, not climb to v4 +0.238')
    print('=' * 78, flush=True)

    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL, 'blocks': BLOCKS,
                      'out_dim': OUT_DIM},
              open(_OUT / 'config.json', 'w'), indent=2)
    assert len(set(HELD_OUT) & set(TRAIN)) == 0

    print(f'\n[LOAD] {PRETRAINED}', flush=True)
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']
    assert len(dec_model.blocks) == 12
    _ow = getattr(dec_model.out_layer, 'weight', None)
    if _ow is not None:
        assert _ow.shape[0] == 101, f'out_layer out_features={_ow.shape[0]} != 101'
        print(f'[CHECK] out_layer 96 -> {_ow.shape[0]} channels', flush=True)
    else:
        print('[CHECK] out_layer exposes no .weight — channel count is asserted '
              'inside the forward hook instead', flush=True)

    for p in flow_model.parameters(): p.requires_grad_(False)
    for p in dec_model.parameters():  p.requires_grad_(False)

    print(f'\n[STRUCT] seed={STRUCT_SEED}  (frame 75, reused by all 150 frames)', flush=True)
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}', flush=True)
    assert N_vox == 7301, f'N_vox={N_vox} != 7301'
    # kept on CPU so the SLaT cache can be checked against it below — a count
    # match is NOT proof the cache came from this seed and this frame.
    fresh_coords = coords.detach().cpu().int().clone()
    del cond_struct, coords; gc.collect(); torch.cuda.empty_cache()

    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    lora = OutLayerLoRA(rank=args.rank).to(DEVICE)
    n_params = sum(p.numel() for p in lora.parameters())
    assert n_params == args.rank * (DEC_OUT_IN_DIM + OUT_DIM), \
        f'param count {n_params} != {args.rank * (DEC_OUT_IN_DIM + OUT_DIM)}'
    print(f'\n[LORA] rank={args.rank}  out_dim={OUT_DIM}  params={n_params:,}  '
          f'(rung8 was {args.rank * (DEC_OUT_IN_DIM + COLOR_DIM):,})')
    print(f'[GATE 0] param count PASSED\n', flush=True)

    # NOTE: rung8 re-encoded all 150 DINO frames and drew a fixed_noise_feats
    # vector here, then used neither — stages A/B/C are served entirely from
    # slat_cache.npz. Dropped in rung9: it cost ~3 min of GPU on every requeue
    # and rung9 will requeue (30 epochs against a hard 4 h partition cap).
    # Nothing downstream reads them, and lora.A is already initialised above, so
    # the RNG position that matters is unchanged from rung8.
    torch.manual_seed(args.seed)

    all_frames       = list(range(1, N_FRAMES + 1))
    _slat_cache_disk = _OUT / 'slat_cache.npz'
    _v4_cache = (_LORA / 'runs' /
                 f'rung5_colonly_outlayer_r4_s6_{args.v4_run_id}' / 'slat_cache.npz')

    if _slat_cache_disk.exists():
        src = _slat_cache_disk
    elif _v4_cache.exists():
        src = _v4_cache
    else:
        raise FileNotFoundError(
            f'No SLaT cache at {_v4_cache}. rung9 must read the SAME cache rung8 '
            f'read, or the comparison is not frame-for-frame.')
    print(f'\n[SLAT CACHE] Loading: {src}', flush=True)
    _npz      = np.load(src)
    _sarr     = _npz['slats']
    _coords_t = torch.from_numpy(_npz['coords']).int()
    assert _sarr.shape[1] == N_vox, \
        f'cache N_vox={_sarr.shape[1]} != freshly sampled {N_vox}'

    # ── PROVENANCE CHECK ──────────────────────────────────────────────────────
    # A matching voxel COUNT is not proof the cache was built from frame 75 at
    # STRUCT_SEED=42. render_pinned_vs_free.py silently compared a seed-42 cache
    # against a seed-6 live sample for exactly this reason: nothing asserted the
    # two structures were the same one. Assert the voxels themselves.
    if _coords_t.shape != fresh_coords.shape or not torch.equal(_coords_t, fresh_coords):
        n_diff = int((_coords_t != fresh_coords).any(dim=1).sum()) \
            if _coords_t.shape == fresh_coords.shape else -1
        raise AssertionError(
            f'SLaT cache coords do NOT match a fresh sample of frame 75 at '
            f'STRUCT_SEED={STRUCT_SEED}. cache={tuple(_coords_t.shape)} '
            f'fresh={tuple(fresh_coords.shape)} differing_voxels={n_diff}. '
            f'The cache was built from a different frame or a different seed — '
            f'training on it would make "pinned coords" mean something other '
            f'than what this run claims.')
    print(f'[PROVENANCE] cache coords == fresh frame-75 sample at seed '
          f'{STRUCT_SEED}  ({N_vox:,} voxels, exact match)', flush=True)

    slat_cache = {fi: (torch.from_numpy(_sarr[fi - 1]).float(), _coords_t)
                  for fi in range(1, N_FRAMES + 1)}
    flow_model.cpu()
    if src is not _slat_cache_disk and not _slat_cache_disk.exists():
        np.savez_compressed(_slat_cache_disk, slats=_sarr, coords=_npz['coords'])
        print(f'[SLAT CACHE] Copied to {_slat_cache_disk}', flush=True)
    print(f'[SLAT CACHE] frames={_sarr.shape[0]}  N_vox={_sarr.shape[1]}', flush=True)

    gc.collect(); torch.cuda.empty_cache()
    dec_model.to(DEVICE)

    print('\n[LPIPS] Loading AlexNet...', flush=True)
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)

    static_rmask = load_render_mask()
    renderer     = make_renderer()

    if args.loss_norm == 'v4compat':
        _n = []
        for fi in range(1, N_FRAMES + 1, 15):
            _, gm = load_gt_v4mask(fi)
            _n.append(float((static_rmask | gm).sum()))
        NORM_PX = float(np.mean(_n))
        print(f'\n[NORM] loss_norm=v4compat  denominator={NORM_PX:,.0f} px '
              f'— gradient scale matched to rung8/v4', flush=True)
    else:
        NORM_PX = None
        print(f'\n[NORM] loss_norm=intersection  denominator = |A| per frame '
              f'(~28,000) — roughly 9.5x; LR may need scaling', flush=True)

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None

    if not resumed:
        print('\n[GATE-plain] B=0 identity check...', flush=True)
        slat_ref = _slat_from_cache(slat_cache, 75)
        with torch.no_grad():
            meshes_frz = dec_model(slat_ref)
        v_frz = meshes_frz[0].vertices.detach().clone()
        c_frz, _ = render_mesh(filter_degenerate_faces(meshes_frz[0]), renderer)
        c_frz = c_frz.detach().clone()
        mesh_co  = geomcol_forward(dec_model, lora, slat_ref, with_grad=False)
        c_co, _  = render_mesh(mesh_co, renderer)
        diff = (c_frz - c_co.detach()).abs().max().item()
        print(f'  max |frozen - lora(B=0)| = {diff:.3e}  (tol=1e-2)', flush=True)
        assert diff < 1e-2, f'GATE-plain FAILED: {diff:.3e}'
        print('[GATE-plain] PASSED', flush=True)
        del meshes_frz, c_frz, mesh_co, c_co
        gc.collect(); torch.cuda.empty_cache()

        # ── GATE-geom: the premise of rung9. Do the geometry channels actually
        # move vertices? Perturb only the geometry rows of B, decode, compare.
        # Skipped for --write-blocks color, which has no geometry rows.
        if args.write_blocks == 'color':
            print('\n[GATE-geom] SKIPPED — write_blocks=color has no geometry rows',
                  flush=True)
        else:
            print('\n[GATE-geom] do sdf/deform deltas move the mesh?', flush=True)
            # Swept, not single-shot. The decoder runs in fp16, where 1.0 + 1e-4
            # rounds straight back to 1.0 — a perturbation small enough to be
            # erased by the forward dtype would fail this gate for a reason that
            # has nothing to do with whether the write window reaches geometry.
            # The smallest scale that moves the mesh IS the fp16 resolution floor
            # for this hook, and it is worth knowing before reading ||B|| curves.
            moved, how, floor = False, 'no scale moved the mesh', None
            for scale in (1e-3, 1e-2, 1e-1, 1.0):
                with torch.no_grad():
                    lora.B.data.zero_()
                    torch.manual_seed(1234)            # same draw at every scale
                    for (s, e), off in zip(BLOCKS, _OFFS):
                        lo, hi = max(s, COLOR_START), min(e, COLOR_END)
                        if lo >= hi:                   # pure geometry block
                            lora.B.data[off:off + (e - s)].normal_(0.0, scale)
                        elif lo > s:                   # all101: rows before 53
                            lora.B.data[off:off + (lo - s)].normal_(0.0, scale)
                    mesh_p = geomcol_forward(dec_model, lora, slat_ref, with_grad=False)
                    v_p = mesh_p.vertices.detach()
                    if v_p.shape[0] != v_frz.shape[0]:
                        moved = True
                        how = (f'vertex count {v_frz.shape[0]:,} -> {v_p.shape[0]:,} '
                               f'(topology responded)')
                    else:
                        d = (v_p - v_frz).norm(dim=1)
                        moved = bool(d.mean().item() > 1e-8)
                        how = (f'mean |dv|={d.mean().item():.3e}  '
                               f'max={d.max().item():.3e}')
                    del mesh_p
                gc.collect(); torch.cuda.empty_cache()
                print(f'  sigma(B_geom)={scale:.0e}  ->  '
                      f'{"MOVED" if moved else "no change"}   {how}', flush=True)
                if moved:
                    floor = scale
                    break
            with torch.no_grad():
                lora.B.data.zero_()
            assert moved, (
                'GATE-geom FAILED: sdf/deform deltas up to sigma=1.0 did not move '
                'the mesh at all. The write window is not reaching geometry — the '
                'block offsets or the channel layout are wrong for this checkpoint.')
            print(f'[GATE-geom] PASSED at sigma={floor:.0e}  (B reset to zero)',
                  flush=True)
            json.dump({'fp16_geom_floor_sigma': floor, 'detail': how},
                      open(_LOGD / 'gate_geom.json', 'w'), indent=2)
            gc.collect(); torch.cuda.empty_cache()

        del v_frz, slat_ref
        gc.collect(); torch.cuda.empty_cache()

        print('\n[GATE-region] silhouette overlap on 10 frames...', flush=True)
        tA, tB, iou_r = measure_regions(dec_model, lora, slat_cache, renderer,
                                        list(range(1, 151, 15)))
        frac = tB / max(tA + tB, 1)
        print(f'  |A| intersection = {tA:,}')
        print(f'  |B| render-only  = {tB:,}   ({frac*100:.1f}% of rendered pixels)')
        print(f'  silhouette IoU   = {iou_r:.4f}')
        assert frac > 0.02, (
            f'region B is only {frac*100:.2f}% of pixels — the mismatched-rim '
            f'premise does not hold; investigate before training.')
        print('[GATE-region] PASSED', flush=True)
        json.dump({'A': tA, 'B': tB, 'frac_B': frac, 'iou': iou_r},
                  open(_LOGD / 'region_stats.json', 'w'), indent=2)

        # The per-epoch ratio is measured on GEOM_FRAMES (10 frames, stride 15).
        # The 0.063 figure in the log came from measure_geometry_change.py at
        # stride 10 (16 frames). Different frame sets, so 0.063 is NOT the right
        # comparator for the per-epoch number — measure the frozen ratio on THESE
        # frames, with B still zero, and compare against that.
        print('\n[BASELINE] frozen-decoder tracking ratio on GEOM_FRAMES...', flush=True)
        assert lora.B.abs().max().item() == 0.0, \
            'baseline must be measured with B=0 or it is not the frozen decoder'
        base_geom = measure_geometry_tracking(dec_model, lora, slat_cache,
                                              renderer, GEOM_FRAMES)
        BASE_RATIO = base_geom['tracking_ratio']
        print(f'  render spread={base_geom["spread_render_pct"]:.2f}%  '
              f'GT spread={base_geom["spread_gt_pct"]:.2f}%  '
              f'ratio={BASE_RATIO:.4f}', flush=True)
        print(f'  B=0, so this IS the frozen decoder. Every per-epoch ratio below '
              f'is measured against THIS number, not against 0.063.', flush=True)
        json.dump(base_geom, open(_LOGD / 'baseline_geometry.json', 'w'), indent=2)
    else:
        print('\n[GATES] SKIPPED — resuming', flush=True)
        # BASE_RATIO must survive a requeue or the epoch line below would
        # NameError on the first resumed epoch.
        _bg = _LOGD / 'baseline_geometry.json'
        BASE_RATIO = json.load(open(_bg))['tracking_ratio'] if _bg.exists() else float('nan')
        print(f'[BASELINE] reloaded frozen ratio = {BASE_RATIO:.4f}', flush=True)

    trainable   = list(lora.parameters())
    optimizer   = torch.optim.Adam(trainable, lr=LR, eps=1e-16, weight_decay=0.0)
    start_epoch = 1
    history     = []
    best_psnr   = -float('inf')

    if resumed:
        print(f'\n[RESUME] {latest_ckpt.name}', flush=True)
        ck = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        assert ck.get('run_id') == RUN_ID, f'run_id mismatch: {ck.get("run_id")} != {RUN_ID}'
        lora.load_state_dict(ck['lora_state'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        best_psnr   = ck.get('best_psnr', -float('inf'))
        loss_scale  = ck.get('loss_scale', LOSS_SCALE0)
        hp = _OUT / 'loss_history.json'
        if hp.exists():
            history = json.load(open(hp))
        print(f'  resumed from epoch {ck["epoch"]}', flush=True)
    else:
        print('\n[RESUME] No checkpoint — starting fresh.', flush=True)

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    if not resumed:
        print('\n[GATE 2] Gradient flow check...', flush=True)
        dec_model.eval()
        slat_75 = _slat_from_cache(slat_cache, 75)
        optimizer.zero_grad()
        mesh, raw_b, frz_b = geomcol_forward(dec_model, lora, slat_75,
                                             with_grad=True, return_feats=True)
        colour, live_rm = render_mesh(mesh, renderer)
        gt_75, gm_75 = load_gt(75)
        _, _, l2, npx = intersection_loss(colour, gt_75, live_rm, gm_75,
                                          lpips_fn, NORM_PX)
        reg = LAMBDA_REG * masked_reg(raw_b, frz_b)
        ((l2 + reg) * loss_scale).backward()
        gB = lora.B.grad
        print(f'  supervised pixels = {npx:,.0f}', flush=True)
        print(f'  B.grad norm = {gB.norm().item():.3e}   '
              f'A.grad norm = {lora.A.grad.norm().item():.3e}', flush=True)
        assert gB is not None and gB.norm().item() > 0, 'GATE 2 FAILED (B)'

        # per-block gradient, so a silently-dead geometry half is visible now
        # rather than after 30 epochs of a flat tracking ratio.
        for (s, e), off in zip(BLOCKS, _OFFS):
            gn = gB[off:off + (e - s)].norm().item()
            print(f'    block [{s}:{e}]  |dB| = {gn:.3e}', flush=True)
            assert gn > 0, (f'GATE 2 FAILED: block [{s}:{e}] receives zero '
                            f'gradient — that half of the adapter is dead.')
        optimizer.zero_grad()
        print('[GATE 2] PASSED\n', flush=True)
        del mesh, colour, gt_75, gm_75, l2, reg, raw_b, frz_b, live_rm, slat_75
        gc.collect(); torch.cuda.empty_cache()

    dec_model.eval()
    baseline_cache = {}
    csv_path = _LOGD / 'epoch_metrics.csv'
    if not csv_path.exists():
        csv_path.write_text('epoch,loss_total,loss_mse,loss_lpips,loss_reg,'
                            'held_psnr_union,held_psnr_inter,held_ssim,held_lpips,'
                            'held_iou,B_norm,B_norm_geom,B_norm_col,dR,dG,dB,'
                            'tracking_ratio,spread_render_pct,verts_mean,'
                            'px_supervised,grad_norm,clip_frac,n_degenerate,time_s\n')

    print(f'[TRAIN] epochs {start_epoch}->{EPOCHS}  frames/epoch={len(TRAIN)}', flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        order = TRAIN[:]
        random.shuffle(order)
        tl = tm = tp = tr = 0.0
        tpx = tg = 0.0
        nclip = n_steps = n_degen = 0

        print(f'\n[EPOCH {epoch}/{EPOCHS}] starting...', flush=True)

        for fi in order:
            slat_norm = _slat_from_cache(slat_cache, fi)
            optimizer.zero_grad()

            mesh, raw_b, frz_b = geomcol_forward(
                dec_model, lora, slat_norm, with_grad=True, return_feats=True)

            # Geometry can now collapse the mesh. Skip the step rather than
            # feeding the renderer an empty mesh and training on a blank image.
            if mesh.faces.shape[0] < 10 or mesh.vertices.shape[0] < 10:
                n_degen += 1
                print(f'  e{epoch:02d} f{fi:03d}  DEGENERATE MESH '
                      f'(V={mesh.vertices.shape[0]} F={mesh.faces.shape[0]}) — step skipped',
                      flush=True)
                del mesh, raw_b, frz_b, slat_norm
                gc.collect(); torch.cuda.empty_cache()
                continue

            colour, live_rm = render_mesh(mesh, renderer)
            gt, gt_mask = load_gt(fi)
            mse, lp, render_loss, npx = intersection_loss(
                colour, gt, live_rm, gt_mask, lpips_fn, NORM_PX)
            reg   = LAMBDA_REG * masked_reg(raw_b, frz_b)
            total = render_loss + reg

            (total * loss_scale).backward()
            for p in trainable:
                if p.grad is not None:
                    p.grad.div_(loss_scale)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP).item()
            if not math.isfinite(gnorm):
                n_degen += 1
                print(f'  e{epoch:02d} f{fi:03d}  NON-FINITE GRAD — step skipped', flush=True)
                optimizer.zero_grad()
                del mesh, colour, gt, gt_mask, mse, lp, render_loss, reg, total
                del raw_b, frz_b, slat_norm, live_rm
                gc.collect(); torch.cuda.empty_cache()
                continue
            optimizer.step()

            tl += total.item(); tm += mse.item(); tp += lp.item()
            tr += float(reg); tpx += npx; n_steps += 1
            tg += gnorm; nclip += int(gnorm > GRAD_CLIP)

            if n_steps % 15 == 0:
                print(f'  e{epoch:02d} [{n_steps:03d}/{len(TRAIN)}] f{fi:03d}  '
                      f'mse={mse.item():.5f}  lpips={lp.item():.5f}  '
                      f'reg={float(reg):.3e}  total={total.item():.5f}  '
                      f'V={mesh.vertices.shape[0]:,}  px={npx:,.0f}', flush=True)

            del mesh, colour, gt, gt_mask, mse, lp, render_loss, reg, total
            del raw_b, frz_b, slat_norm, live_rm
            gc.collect(); torch.cuda.empty_cache()

        held  = evaluate_frames(dec_model, lora, slat_cache, renderer,
                                static_rmask, lpips_fn, HELD_OUT)
        shift = measure_channel_shift(dec_model, lora, slat_cache, renderer, DIAG_FRAMES)
        geom  = measure_geometry_tracking(dec_model, lora, slat_cache,
                                          renderer, GEOM_FRAMES)

        b_all, b_geom, b_col = _b_row_norms(lora)
        ep_t = time.time() - t0
        rec = {
            'epoch': epoch,
            'loss_total': tl / max(n_steps, 1), 'loss_mse': tm / max(n_steps, 1),
            'loss_lpips': tp / max(n_steps, 1), 'loss_reg': tr / max(n_steps, 1),
            'held_psnr': held['psnr_mean'], 'held_std': held['psnr_std'],
            'held_psnr_inter': held['psnr_inter_mean'],
            'held_ssim': held['ssim_mean'], 'held_lpips': held['lpips_mean'],
            'held_iou': held['iou_mean'],
            'B_norm': b_all, 'B_norm_geom': b_geom, 'B_norm_col': b_col,
            'shift': [float(x) for x in shift],
            'tracking_ratio': geom['tracking_ratio'],
            'base_ratio': BASE_RATIO,
            'spread_render_pct': geom['spread_render_pct'],
            'spread_gt_pct': geom['spread_gt_pct'],
            'verts_mean': geom['verts_mean'],
            'verts_spread_pct': geom['verts_spread_pct'],
            'px_supervised': tpx / max(n_steps, 1), 'time_s': ep_t,
            'grad_norm': tg / max(n_steps, 1),
            'clip_frac': nclip / max(n_steps, 1),
            'n_degenerate': n_degen,
        }
        new_best = rec['held_psnr'] > best_psnr
        if new_best:
            best_psnr = rec['held_psnr']

        print(f'\n[EPOCH {epoch}/{EPOCHS}]  loss={rec["loss_total"]:.5f}  '
              f'(mse={rec["loss_mse"]:.5f}  lpips*0.1={rec["loss_lpips"]*W_LPIPS:.5f}  '
              f'reg={rec["loss_reg"]:.3e})', flush=True)
        print(f'  ||B||={b_all:.4f}   geom rows={b_geom:.4f}   colour rows={b_col:.4f}',
              flush=True)
        print(f'  held  PSNR_union={rec["held_psnr"]:.3f}+-{rec["held_std"]:.3f}  '
              f'PSNR_inter={rec["held_psnr_inter"]:.3f}  IoU={rec["held_iou"]:.4f}', flush=True)
        print(f'        SSIM={rec["held_ssim"]:.4f}  LPIPS={rec["held_lpips"]:.4f}', flush=True)
        print(f'  SUCCESS  tracking_ratio={geom["tracking_ratio"]:.4f}   '
              f'(render spread {geom["spread_render_pct"]:.2f}% / '
              f'GT {geom["spread_gt_pct"]:.2f}%)   '
              f'[frozen, same frames: {BASE_RATIO:.4f}  ->  '
              f'{geom["tracking_ratio"] / BASE_RATIO:.2f}x]', flush=True)
        print(f'           verts={geom["verts_mean"]:,.0f} '
              f'(spread {geom["verts_spread_pct"]:.2f}%)', flush=True)
        print(f'  GUARD    dR={shift[0]:+.4f}  dG={shift[1]:+.4f}  dB={shift[2]:+.4f}   '
              f'[rung8: +0.105   v4: +0.238]'
              + ('   <-- dB REGRESSING toward v4' if shift[2] > 0.16 else ''), flush=True)
        print(f'  grad_norm={rec["grad_norm"]:.4f}  clipped={rec["clip_frac"]*100:.1f}% of steps'
              + ('   <-- CLIP SATURATING, effective LR is capped' if rec['clip_frac'] > 0.5 else ''),
              flush=True)
        print(f'  px/step={rec["px_supervised"]:,.0f}  degenerate={n_degen}  '
              f'GPU={torch.cuda.max_memory_allocated()/1e9:.1f}GB  '
              + ('BEST ' if new_best else '') + f' time={ep_t:.1f}s', flush=True)

        # LOGS FIRST, THEN CHECKPOINT. If preempted between the two, resume
        # starts at epoch+1 and this epoch's row would be lost forever. Writing
        # logs first makes the worst case a duplicate row, which is recoverable.
        history = [h for h in history if h['epoch'] != epoch]
        history.append(rec)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)
        with open(csv_path, 'a') as fh:
            fh.write(f'{epoch},{rec["loss_total"]:.6f},{rec["loss_mse"]:.6f},'
                     f'{rec["loss_lpips"]:.6f},{rec["loss_reg"]:.6e},'
                     f'{rec["held_psnr"]:.4f},{rec["held_psnr_inter"]:.4f},'
                     f'{rec["held_ssim"]:.4f},{rec["held_lpips"]:.4f},'
                     f'{rec["held_iou"]:.4f},'
                     f'{b_all:.4f},{b_geom:.4f},{b_col:.4f},'
                     f'{shift[0]:.5f},{shift[1]:.5f},{shift[2]:.5f},'
                     f'{rec["tracking_ratio"]:.5f},{rec["spread_render_pct"]:.4f},'
                     f'{rec["verts_mean"]:.0f},'
                     f'{rec["px_supervised"]:.0f},{rec["grad_norm"]:.5f},'
                     f'{rec["clip_frac"]:.4f},{n_degen},{ep_t:.1f}\n')
        save_curves(history)

        _ck = {'epoch': epoch, 'run_id': RUN_ID, 'best_psnr': best_psnr,
               'loss_scale': loss_scale, 'optimizer': optimizer.state_dict(),
               'lora_state': lora.state_dict(),
               'write_blocks': args.write_blocks, 'blocks': BLOCKS}
        torch.save(_ck, _CKPT / f'lora_e{epoch:03d}.pt')
        if new_best:
            torch.save(_ck, _CKPT / 'lora_best.pt')
            print('  [CKPT] new best -> lora_best.pt', flush=True)

        if epoch % DIAG_EVERY == 0:
            ed = _DIAG / f'e{epoch:03d}'; ed.mkdir(exist_ok=True)
            for fi in DIAG_FRAMES:
                slat = _slat_from_cache(slat_cache, fi)
                mesh = geomcol_forward(dec_model, lora, slat, with_grad=False)
                c, _ = render_mesh(mesh, renderer)
                r = c.detach().clamp(0, 1)
                if fi not in baseline_cache:
                    with torch.no_grad():
                        mb = dec_model(slat)
                    cb, _ = render_mesh(mb[0], renderer)
                    baseline_cache[fi] = cb.detach().clamp(0, 1)
                make_strip([
                    (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
                    (baseline_cache[fi], 'frozen decoder'),
                    (r, f'rung9 {args.write_blocks} e{epoch:03d}'),
                ]).save(ed / f'strip_f{fi:04d}.png')
                del r, mesh, c, slat
                gc.collect(); torch.cuda.empty_cache()

    print('\n' + '=' * 78)
    print('[FINAL EVAL] All 150 frames')
    print('=' * 78, flush=True)

    bp = _CKPT / 'lora_best.pt'
    if bp.exists():
        bc = torch.load(bp, map_location=DEVICE, weights_only=True)
        lora.load_state_dict(bc['lora_state'])
        print(f'  Loaded best (epoch={bc["epoch"]}  psnr={bc["best_psnr"]:.3f})', flush=True)

    res_held  = evaluate_frames(dec_model, lora, slat_cache, renderer, static_rmask, lpips_fn, HELD_OUT)
    res_train = evaluate_frames(dec_model, lora, slat_cache, renderer, static_rmask, lpips_fn, TRAIN)
    res_all   = evaluate_frames(dec_model, lora, slat_cache, renderer, static_rmask, lpips_fn, all_frames)
    final_shift = measure_channel_shift(dec_model, lora, slat_cache, renderer,
                                        list(range(1, 151, 10)))

    print(f'\n  METRICS ON v4\'s REGION (comparable to v4 and rung8):')
    print(f'    All:   PSNR={res_all["psnr_mean"]:.3f}+-{res_all["psnr_std"]:.3f}   (v4: 10.647)')
    print(f'    Held:  PSNR={res_held["psnr_mean"]:.3f}+-{res_held["psnr_std"]:.3f}   (v4: 10.664)')
    print(f'    Train: PSNR={res_train["psnr_mean"]:.3f}+-{res_train["psnr_std"]:.3f}   (v4: 10.645)')
    print(f'    SSIM={res_all["ssim_mean"]:.4f} (v4: 0.8671)   '
          f'LPIPS={res_all["lpips_mean"]:.4f} (v4: 0.1256)   '
          f'IoU={res_all["iou_mean"]:.4f} (frozen: 0.76)')
    print(f'\n  METRICS ON THE INTERSECTION (do NOT compare to v4):')
    print(f'    All:   PSNR={res_all["psnr_inter_mean"]:.3f}')

    print(f'\n  COLOUR GUARD  (15 frames):')
    print(f'    dR={final_shift[0]:+.4f}  dG={final_shift[1]:+.4f}  dB={final_shift[2]:+.4f}')
    print(f'    rung8 was +0.105 on dB;  v4 was +0.238')
    guard = 'HELD' if final_shift[2] < 0.16 else 'BROKEN'
    print(f'    -> colour guard {guard}')

    print('\n' + '=' * 78)
    print('[GEOMETRY] Full diagnostic — 4 orbit views + chamfer')
    print('=' * 78, flush=True)
    ratio, verdict = final_geometry_report(dec_model, lora, slat_cache, renderer,
                                           _OUT / 'geometry_report.json')

    json.dump({'run_id': RUN_ID, 'write_blocks': args.write_blocks,
               'held': res_held, 'train': res_train, 'all': res_all,
               'final_shift': [float(x) for x in final_shift],
               'rung8_shift_dB': 0.105, 'v4_shift': [0.2457, 0.2233, 0.2380],
               'colour_guard': guard,
               'tracking_ratio': ratio, 'baseline_frozen_ratio': 0.063,
               'geometry_verdict': verdict},
              open(_OUT / 'final_eval.json', 'w'), indent=2)

    print(f'\n[SUMMARY] tracking_ratio={ratio:.3f} (baseline 0.063)   '
          f'colour guard {guard}   best held PSNR={best_psnr:.3f} dB')
    print(f'[DONE]  output={_OUT}', flush=True)


if __name__ == '__main__':
    main()
