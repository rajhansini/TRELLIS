"""
rung8_intersection_lora.py
--------------------------
v4 out_layer colour LoRA, trained on the INTERSECTION of the two silhouettes
instead of their union. Fixes the white cast.

WHY
  TRELLIS generates its own teapot; the Kling video has a different one. Their
  silhouettes overlap only ~76%. In v4 the loss covered the UNION, so ~16% of
  the rendered teapot's pixels sat on white GT background and were supervised
  toward white. The residual there is (dark lava - white), the largest possible,
  so that thin rim supplied ~56% of the entire colour gradient, sign [+,+,+] on
  every channel. The adapter reads voxel FEATURES, not position, so it cannot
  learn "be white only at the rim" — it learns "be whiter" as a function of
  appearance, and the whole body washes out. Measured: blue 4.5 -> 68/255 when
  the target was 27/255.

  Modelling the adapter as a global additive shift (its first-order behaviour,
  since it has no positional input), least squares over each region predicts:
      union         d* = [sum_A(g-f) + sum_B(1-f)] / (|A|+|B|)
      intersection  d* =  sum_A(g-f) / |A|
  Measured against v4's actual output over all 150 frames: union error
  0.027+-0.014, intersection error 0.111+-0.019, union closer on 149/150 frames,
  paired t = 40.0. v4 did exactly what the union region demanded.

  Dropping the rim is justified because GEOMETRY IS FROZEN: the silhouette
  cannot change whatever the adapter learns, so supervising the mismatch can
  never fix the shape — it can only corrupt the colour.

CHANGES vs rung5_colonly_lora_v4.py  (four)
  1. loss region  (render_mask | gt_mask)  ->  (render_mask & gt_mask)
  2. the render silhouette used is the LIVE per-frame mask from the renderer,
     not the single static mask.png reused for all 150 frames (v4 computed the
     live mask and then threw it away)
  3. lambda_reg defaults to 0.0, stated explicitly. v4's config said 0.01 but a
     tensor-view aliasing bug made it a no-op (its loss_history shows reg ~1e-13
     against mse ~1.7e-2 for all 30 epochs), so v4 was effectively unregularised.
     Defaulting to 0 reproduces that honestly and keeps this a ONE-VARIABLE
     change. The reg code itself is fixed here (frozen_col is .clone()d), so
     passing --lambda-reg > 0 gives a live regulariser for a later run.
  4. gt_mask is min(RGB) < 0.95 instead of (gt < 0.99).any(). REQUIRED, not
     cosmetic: the Kling background is off-white (per-channel corner minimum
     [0.9843, 0.9882, 0.9765]) so v4's test fired on the background and returned
     99.7% of the image. Intersecting with that drops 437 px of 34,416 — the fix
     would have been a no-op. With the tight mask it drops 18.2%, and those
     dropped pixels have mean GT value 0.988, i.e. exactly the white-targeting
     set. See logs/gt_mask_threshold.log and logs/rung8_audit.log.
     The loss normaliser is held at v4's count so gradient scale is unchanged.

Everything else — 576 params, rank 4, out_layer hook, lr, seed, epochs, clamp,
held-out split, LOSS_SCALE — is identical to v4.

EVALUATION is deliberately UNCHANGED from v4: PSNR/SSIM/LPIPS are reported over
v4's region using the static mask.png, so the numbers are directly comparable.
Intersection-region metrics are logged alongside as a secondary column — do not
compare those to v4, they are measured on an easier region.

SUCCESS CRITERION, stated in advance
  mean blue-channel shift (lora - frozen) over the rendered object should land
  near +0.079, not v4's +0.238. Logged every epoch as dB. ||B|| should plateau
  rather than climbing to the final epoch as it did in v4 (0.13 -> 1.65,
  monotonic, still rising at epoch 30).

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  python -u experiments/lora_experiments/visibility/rung8_intersection_lora.py \
    --rank 4 --epochs 30 --seed 6 \
    2>&1 | tee experiments/lora_experiments/visibility/logs/rung8.log
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
                  help='0.0 reproduces v4 effective behaviour (its reg was a '
                       'no-op). >0 enables the FIXED regulariser.')
_ap_.add_argument('--smoke',      action='store_true')
_ap_.add_argument('--diag-every', type=int,   default=5)
_ap_.add_argument('--loss-norm',  default='v4compat',
                  choices=['v4compat', 'intersection'],
                  help="v4compat divides by v4's pixel count so the gradient "
                       "scale is unchanged (true one-variable ablation). "
                       "intersection is a proper mean, ~9.5x the gradient.")
_ap_.add_argument('--v4-run-id',  default='c85c888f',
                  help='v4 run whose slat_cache.npz is reused')
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
COLOR_START    = 53
COLOR_END      = 101
COLOR_DIM      = COLOR_END - COLOR_START   # 48
ABS_MIN        = args.abs_min
ABS_MAX        = args.abs_max
LAMBDA_REG     = args.lambda_reg

HELD_OUT = list(range(5, 151, 10))
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

_CFG = dict(
    variant='color_only_outlayer_intersection',
    loss_region='intersection(live_render_mask & gt_mask)',
    rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    abs_min=ABS_MIN, abs_max=ABS_MAX, lambda_reg=LAMBDA_REG,
    gt_bg_thresh=0.95, loss_norm=args.loss_norm, held_out=HELD_OUT,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = f'rung8_intersect_r{args.rank}_s{args.seed}_{RUN_ID}'
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


class OutLayerLoRA(nn.Module):
    """Byte-identical to v4 so checkpoints are interchangeable."""
    def __init__(self, rank: int, in_dim: int = DEC_OUT_IN_DIM,
                 color_dim: int = COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def colonly_forward(dec_model, lora, slat_norm, with_grad=True, return_feats=False):
    captured = {}

    def _hook(mod, inp, out):
        x_feats = inp[0].feats
        delta   = lora(x_feats)
        new_feats = out.feats.clone()
        # .clone() is load-bearing: new_feats[:, C0:C1] is a VIEW, and the
        # in-place write below would otherwise overwrite the captured "frozen"
        # colours, making the regulariser evaluate (raw - clamp(raw)) ~ 0.
        # That is the v4 bug that made lambda_reg a no-op.
        frozen_col = new_feats[:, COLOR_START:COLOR_END].clone()
        raw_col    = frozen_col + delta
        captured['frozen_col'] = frozen_col.detach()
        captured['raw_col']    = raw_col
        new_feats[:, COLOR_START:COLOR_END] = raw_col.clamp(ABS_MIN, ABS_MAX)
        return out.replace(new_feats)

    _ctx = torch.no_grad() if not with_grad else _nullctx()
    with _ctx:
        h = dec_model.out_layer.register_forward_hook(_hook)
        meshes = dec_model(slat_norm)
        h.remove()

    mesh = meshes[0]
    if return_feats:
        return mesh, captured['raw_col'], captured['frozen_col']
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
    """
    gt_mask must be the GT TEAPOT, not "anything not pure white".

    v4 used (gt < 0.99).any(dim=0). The Kling frames have an OFF-WHITE
    background — measured corner minimum per channel [0.9843, 0.9882, 0.9765] —
    so that test fires on the background too and returns 99.7% of the image.
    Under a union loss that only dilutes the normaliser, but it would have made
    an intersection loss a no-op: live_rm & (almost everything) == live_rm, rim
    and all.

    min(RGB) < 0.95 gives 11.5% object with 0.00% background leak, and is flat
    from 0.97 down to 0.90, so it is not sitting on a knife edge.
    See logs/gt_mask_threshold.log.
    """
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


# ── THE CHANGE ────────────────────────────────────────────────────────────────

def intersection_loss(rendered, gt, live_rmask, gt_mask, lpips_fn, norm_px=None):
    """
    Supervise ONLY where the rendered teapot and the GT teapot overlap.

    v4 used   m = (static_render_mask | gt_mask)   <- union, includes the rim
    here      m = (live_render_mask   & gt_mask)   <- intersection

    live_rmask is the renderer's own per-frame silhouette. v4 computed it and
    discarded it, using a single static mask.png for all 150 frames instead.

    NORMALISER. v4 divided by |static_mask | leaky_gt_mask| ~ 268,000 (the leaky
    GT mask made that nearly the whole image) while only ~34,000 pixels actually
    carried gradient. A proper mean over the intersection (~28,000 px) is 9.5x
    larger for the same per-pixel error, so at the same LR=1e-4 and GRAD_CLIP=1.0
    the clip would saturate and the effective step size would change — that is a
    second variable, and the comparison to v4 would no longer be controlled.

      --loss-norm v4compat     divide by v4's count. Gradient scale preserved,
                               so the ONLY change is which pixels contribute.
                               This is the ablation. DEFAULT.
      --loss-norm intersection proper mean over supervised pixels. Correct in
                               isolation, but ~9.5x the gradient; use with a
                               correspondingly smaller LR.

    Returns (mse, lpips, total, n_pixels_supervised).
    """
    m = (live_rmask & gt_mask).float()
    n = m.sum()
    if n < 1.0:                                   # degenerate; should never fire
        z = rendered.sum() * 0.0
        return z, z, z, 0.0
    denom = (norm_px if norm_px is not None else n)
    mse = ((rendered - gt) ** 2 * m).sum() / (denom * 3 + 1e-8)
    r   = rendered * m + (1 - m)                  # composite outside onto white
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


def masked_reg(raw_col, frozen_col):
    return (raw_col - frozen_col).pow(2).mean()


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl):
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(ns, t_ten, cond_gl)
            ns    = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def render_mesh(mesh, renderer):
    """Returns (composited colour, LIVE boolean silhouette)."""
    ext  = EXTRINSICS.to(DEVICE)
    intr = INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    m    = res['mask']
    colour = res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))
    return colour, (m > 0.5)


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
        except: tw = len(lbl) * 7
        draw.text((col * cell + (cell - tw) // 2, 6), lbl, fill=(210, 210, 210))
    return canvas


def measure_channel_shift(dec_model, lora, slat_cache, renderer, frames):
    """
    THE SUCCESS CRITERION, measured every epoch.

    mean(lora - frozen) per RGB channel over the rendered object.
    Measured with the SAME tight GT mask this run trains on:
      v4 reached           [+0.2457, +0.2233, +0.2380]
      intersection predicts[+0.1572, +0.0853, +0.0793]  <- target
    """
    dec_model.eval()
    d = torch.zeros(3, device=DEVICE); n = 0
    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = colonly_forward(dec_model, lora, slat, with_grad=False)
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


def measure_regions(dec_model, lora, slat_cache, renderer, frames, static_rmask):
    """Log |A|, |B| and silhouette IoU so the premise is auditable per run."""
    dec_model.eval()
    tA = tB = tU = 0
    with torch.no_grad():
        for fi in frames:
            slat = _slat_from_cache(slat_cache, fi)
            mesh = colonly_forward(dec_model, lora, slat, with_grad=False)
            _, rm = render_mesh(mesh, renderer)
            _, gm = load_gt(fi)
            A = (rm & gm).sum().item()
            B = (rm & ~gm).sum().item()
            U = (rm | gm).sum().item()
            tA += A; tB += B; tU += U
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
        mesh  = colonly_forward(dec_model, lora, slat_norm, with_grad=False)
        colour, live_rm = render_mesh(mesh, renderer)
        render = colour.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)                 # tight teapot mask
        _,  gm_v4   = load_gt_v4mask(fi)           # v4's leaky mask, for parity
        psnr_u = masked_psnr(render, gt, static_rmask)          # v4-comparable
        inter  = (live_rm & gt_mask)
        psnr_i = masked_psnr(render, gt, inter) if inter.sum() > 0 else float('nan')
        ssim   = compute_ssim(render, gt)
        _, lp  = union_metrics(render, gt, static_rmask, gm_v4, lpips_fn)
        per_frame.append({'frame': fi, 'psnr': psnr_u, 'psnr_inter': psnr_i,
                          'ssim': ssim, 'lpips': lp})
        del render, gt, gt_mask, gm_v4, mesh, colour, live_rm
        gc.collect(); torch.cuda.empty_cache()
    f = lambda k: [r[k] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(f('psnr'))), 'psnr_std': float(np.std(f('psnr'))),
        'psnr_inter_mean': float(np.nanmean(f('psnr_inter'))),
        'ssim_mean': float(np.mean(f('ssim'))), 'ssim_std': float(np.std(f('ssim'))),
        'lpips_mean': float(np.mean(f('lpips'))), 'lpips_std': float(np.std(f('lpips'))),
        'per_frame': per_frame,
    }


def save_curves(history):
    if len(history) < 2:
        return
    ep  = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Rung 8 intersection loss — {_LABEL}', fontsize=11)
    axes[0].plot(ep, [r['loss_mse'] for r in history], 'o-', color='#e06c75', label='MSE')
    axes[0].plot(ep, [r['loss_lpips'] for r in history], 's-', color='#d19a66', label='LPIPS')
    axes[0].plot(ep, [r['loss_total'] for r in history], '^-', color='#c678dd', label='total')
    axes[0].set_title('Train loss (intersection)'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, [r['held_psnr'] for r in history], 'o-', color='#98c379', label='union (v4-comparable)')
    axes[1].plot(ep, [r['held_psnr_inter'] for r in history], 's--', color='#56b6c2', label='intersection')
    axes[1].set_title('Held-out PSNR (dB)'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, [r['B_norm'] for r in history], 'o-', color='#61afef')
    axes[2].axhline(1.6487, color='#e06c75', ls='--', lw=1, label='v4 final ||B||=1.65')
    axes[2].set_title('||B||'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    for i, (c, col) in enumerate(zip('RGB', ('#e06c75', '#98c379', '#61afef'))):
        axes[3].plot(ep, [r['shift'][i] for r in history], 'o-', color=col, label=f'd{c}')
    for y, lbl, st in ((0.238, 'v4 dB = +0.238', '--'), (0.079, 'target dB = +0.079', ':')):
        axes[3].axhline(y, color='#888', ls=st, lw=1, label=lbl)
    axes[3].set_title('Channel shift (lora - frozen)'); axes[3].legend(fontsize=7)
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


def find_latest_ckpt():
    c = sorted(_CKPT.glob('lora_e*.pt'))
    return c[-1] if c else None


def main():
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    print('=' * 78)
    print('Rung 8 — INTERSECTION loss colour LoRA (out_layer hook)')
    print(f'  loss region   : live_render_mask & gt_mask   (v4 used union)')
    print(f'  render mask   : LIVE per-frame  (v4 used static mask.png)')
    print(f'  rank          : {args.rank}     seed: {args.seed}     epochs: {EPOCHS}')
    print(f'  lambda_reg    : {LAMBDA_REG}  '
          f'({"reproduces v4 effective behaviour" if LAMBDA_REG == 0 else "FIXED reg active"})')
    print(f'  abs_clamp     : [{ABS_MIN}, {ABS_MAX}]     LOSS_SCALE: {LOSS_SCALE0:.0f}')
    print(f'  run_id        : {RUN_ID}')
    print(f'  output        : {_OUT}')
    print(f'  train.log     : {_OUT / "train.log"}')
    print('  SUCCESS CRITERION: dB (blue shift) -> ~+0.10, not v4 +0.25')
    print('=' * 78, flush=True)

    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL},
              open(_OUT / 'config.json', 'w'), indent=2)
    assert len(set(HELD_OUT) & set(TRAIN)) == 0

    print(f'\n[LOAD] {PRETRAINED}', flush=True)
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']
    assert len(dec_model.blocks) == 12

    for p in flow_model.parameters(): p.requires_grad_(False)
    for p in dec_model.parameters():  p.requires_grad_(False)

    print(f'\n[STRUCT] seed={STRUCT_SEED}', flush=True)
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}', flush=True)
    assert N_vox == 7301, f'N_vox={N_vox} != 7301'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    lora = OutLayerLoRA(rank=args.rank).to(DEVICE)
    n_params = sum(p.numel() for p in lora.parameters())
    assert n_params == args.rank * (DEC_OUT_IN_DIM + COLOR_DIM)
    print(f'\n[LORA] rank={args.rank}  params={n_params:,}')
    print(f'[GATE 0] param count PASSED\n', flush=True)

    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    print(f'[DINO] Encoding {N_FRAMES} frames...', flush=True)
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}', flush=True)
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

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
            f'No SLaT cache at {_v4_cache}. Reusing v4\'s cache is required so '
            f'this run is comparable to v4 frame for frame.')
    print(f'\n[SLAT CACHE] Loading: {src}', flush=True)
    _npz      = np.load(src)
    _sarr     = _npz['slats']
    _coords_t = torch.from_numpy(_npz['coords']).int()
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

    # v4's normaliser: |static_rmask | leaky_gt_mask|, averaged over frames.
    if args.loss_norm == 'v4compat':
        _n = []
        for fi in range(1, N_FRAMES + 1, 15):
            _, gm = load_gt_v4mask(fi)
            _n.append(float((static_rmask | gm).sum()))
        NORM_PX = float(np.mean(_n))
        print(f'\n[NORM] loss_norm=v4compat  denominator={NORM_PX:,.0f} px '
              f'(v4\'s count) — gradient scale matched to v4', flush=True)
    else:
        NORM_PX = None
        print(f'\n[NORM] loss_norm=intersection  denominator = |A| per frame '
              f'(~28,000) — roughly 9.5x v4 gradient; LR may need scaling',
              flush=True)

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None

    if not resumed:
        print('\n[GATE-plain] B=0 identity check...', flush=True)
        slat_ref = _slat_from_cache(slat_cache, 75)
        with torch.no_grad():
            meshes_frz = dec_model(slat_ref)
        c_frz, _ = render_mesh(filter_degenerate_faces(meshes_frz[0]), renderer)
        mesh_co  = colonly_forward(dec_model, lora, slat_ref, with_grad=False)
        c_co, _  = render_mesh(mesh_co, renderer)
        diff = (c_frz.detach() - c_co.detach()).abs().max().item()
        print(f'  max |frozen - lora(B=0)| = {diff:.3e}  (tol=1e-2)', flush=True)
        assert diff < 1e-2, f'GATE-plain FAILED: {diff:.3e}'
        print('[GATE-plain] PASSED', flush=True)
        del meshes_frz, c_frz, mesh_co, c_co
        gc.collect(); torch.cuda.empty_cache()

        # ── GATE-region: the premise of this whole run, logged as evidence ──
        print('\n[GATE-region] silhouette overlap on 10 frames...', flush=True)
        tA, tB, iou = measure_regions(dec_model, lora, slat_cache, renderer,
                                      list(range(1, 151, 15)), static_rmask)
        frac = tB / max(tA + tB, 1)
        print(f'  |A| intersection = {tA:,}')
        print(f'  |B| render-only  = {tB:,}   ({frac*100:.1f}% of supervised pixels)')
        print(f'  silhouette IoU   = {iou:.4f}')
        print(f'  -> v4 supervised region B toward WHITE; this run excludes it.')
        assert frac > 0.02, (
            f'region B is only {frac*100:.2f}% of pixels — the premise of this run '
            f'(a large mismatched rim) does not hold; investigate before training.')
        print('[GATE-region] PASSED', flush=True)
        json.dump({'A': tA, 'B': tB, 'frac_B': frac, 'iou': iou},
                  open(_LOGD / 'region_stats.json', 'w'), indent=2)
    else:
        print('\n[GATES] SKIPPED — resuming', flush=True)

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
        mesh, raw_lc, frz_c = colonly_forward(dec_model, lora, slat_75,
                                              with_grad=True, return_feats=True)
        colour, live_rm = render_mesh(mesh, renderer)
        gt_75, gm_75 = load_gt(75)
        _, _, l2, npx = intersection_loss(colour, gt_75, live_rm, gm_75,
                                          lpips_fn, NORM_PX)
        reg = LAMBDA_REG * masked_reg(raw_lc, frz_c)
        ((l2 + reg) * loss_scale).backward()
        print(f'  supervised pixels = {npx:,.0f}', flush=True)
        print(f'  B.grad norm = {lora.B.grad.norm().item():.3e}   '
              f'A.grad norm = {lora.A.grad.norm().item():.3e}', flush=True)
        assert lora.B.grad is not None and lora.B.grad.norm().item() > 0, 'GATE 2 FAILED'
        optimizer.zero_grad()
        print('[GATE 2] PASSED\n', flush=True)
        del mesh, colour, gt_75, gm_75, l2, reg, raw_lc, frz_c, live_rm
        gc.collect(); torch.cuda.empty_cache()

    dec_model.eval()
    baseline_cache = {}
    csv_path = _LOGD / 'epoch_metrics.csv'
    if not csv_path.exists():
        csv_path.write_text('epoch,loss_total,loss_mse,loss_lpips,loss_reg,'
                            'held_psnr_union,held_psnr_inter,held_ssim,held_lpips,'
                            'B_norm,dR,dG,dB,px_supervised,grad_norm,clip_frac,time_s\n')

    print(f'[TRAIN] epochs {start_epoch}->{EPOCHS}  frames/epoch={len(TRAIN)}', flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        order = TRAIN[:]
        random.shuffle(order)
        tl = tm = tp = tr = 0.0
        tpx = tg = 0.0
        nclip = 0
        n_steps = 0

        print(f'\n[EPOCH {epoch}/{EPOCHS}] starting...', flush=True)

        for fi in order:
            slat_norm = _slat_from_cache(slat_cache, fi)
            optimizer.zero_grad()

            mesh, raw_col, frozen_col = colonly_forward(
                dec_model, lora, slat_norm, with_grad=True, return_feats=True)
            colour, live_rm = render_mesh(mesh, renderer)
            gt, gt_mask = load_gt(fi)
            mse, lp, render_loss, npx = intersection_loss(
                colour, gt, live_rm, gt_mask, lpips_fn, NORM_PX)
            reg   = LAMBDA_REG * masked_reg(raw_col, frozen_col)
            total = render_loss + reg

            (total * loss_scale).backward()
            for p in trainable:
                if p.grad is not None:
                    p.grad.div_(loss_scale)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP).item()
            optimizer.step()

            tl += total.item(); tm += mse.item(); tp += lp.item()
            tr += float(reg); tpx += npx; n_steps += 1
            tg += gnorm; nclip += int(gnorm > GRAD_CLIP)

            if n_steps % 15 == 0:
                print(f'  e{epoch:02d} [{n_steps:03d}/{len(TRAIN)}] f{fi:03d}  '
                      f'mse={mse.item():.5f}  lpips={lp.item():.5f}  '
                      f'reg={float(reg):.3e}  total={total.item():.5f}  '
                      f'px={npx:,.0f}', flush=True)

            del mesh, colour, gt, gt_mask, mse, lp, render_loss, reg, total
            del raw_col, frozen_col, slat_norm, live_rm
            gc.collect(); torch.cuda.empty_cache()

        held = evaluate_frames(dec_model, lora, slat_cache, renderer,
                               static_rmask, lpips_fn, HELD_OUT)
        shift = measure_channel_shift(dec_model, lora, slat_cache, renderer, DIAG_FRAMES)

        b_norm = lora.B.float().norm().item()
        ep_t   = time.time() - t0
        rec = {
            'epoch': epoch,
            'loss_total': tl / max(n_steps, 1), 'loss_mse': tm / max(n_steps, 1),
            'loss_lpips': tp / max(n_steps, 1), 'loss_reg': tr / max(n_steps, 1),
            'held_psnr': held['psnr_mean'], 'held_std': held['psnr_std'],
            'held_psnr_inter': held['psnr_inter_mean'],
            'held_ssim': held['ssim_mean'], 'held_lpips': held['lpips_mean'],
            'B_norm': b_norm, 'shift': [float(x) for x in shift],
            'px_supervised': tpx / max(n_steps, 1), 'time_s': ep_t,
            'grad_norm': tg / max(n_steps, 1),
            'clip_frac': nclip / max(n_steps, 1),
        }
        new_best = rec['held_psnr'] > best_psnr
        if new_best:
            best_psnr = rec['held_psnr']

        print(f'\n[EPOCH {epoch}/{EPOCHS}]  loss={rec["loss_total"]:.5f}  '
              f'(mse={rec["loss_mse"]:.5f}  lpips*0.1={rec["loss_lpips"]*W_LPIPS:.5f}  '
              f'reg={rec["loss_reg"]:.3e})', flush=True)
        print(f'  ||B||={b_norm:.4f}   (v4 final was 1.6487, still rising)', flush=True)
        print(f'  held  PSNR_union={rec["held_psnr"]:.3f}+-{rec["held_std"]:.3f}  '
              f'(v4: 10.664)   PSNR_inter={rec["held_psnr_inter"]:.3f}', flush=True)
        print(f'        SSIM={rec["held_ssim"]:.4f}  LPIPS={rec["held_lpips"]:.4f}', flush=True)
        print(f'  SHIFT dR={shift[0]:+.4f}  dG={shift[1]:+.4f}  dB={shift[2]:+.4f}   '
              f'[v4: +0.246/+0.223/+0.238   target: +0.157/+0.085/+0.079]', flush=True)
        print(f'  grad_norm={rec["grad_norm"]:.4f}  clipped={rec["clip_frac"]*100:.1f}% of steps'
              + ('   <-- CLIP SATURATING, effective LR is capped' if rec['clip_frac'] > 0.5 else ''),
              flush=True)
        print(f'  px/step={rec["px_supervised"]:,.0f}  '
              f'GPU={torch.cuda.max_memory_allocated()/1e9:.1f}GB  '
              + ('BEST ' if new_best else '') + f' time={ep_t:.1f}s', flush=True)

        # LOGS FIRST, THEN CHECKPOINT. If the job is preempted between the two,
        # resume starts at epoch+1 and the row for this epoch would be lost
        # forever. Writing logs first means the worst case is a duplicate row,
        # which is recoverable, rather than a hole, which is not.
        history = [h for h in history if h['epoch'] != epoch]   # drop dup on resume
        history.append(rec)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)
        with open(csv_path, 'a') as fh:
            fh.write(f'{epoch},{rec["loss_total"]:.6f},{rec["loss_mse"]:.6f},'
                     f'{rec["loss_lpips"]:.6f},{rec["loss_reg"]:.6e},'
                     f'{rec["held_psnr"]:.4f},{rec["held_psnr_inter"]:.4f},'
                     f'{rec["held_ssim"]:.4f},{rec["held_lpips"]:.4f},'
                     f'{b_norm:.4f},{shift[0]:.5f},{shift[1]:.5f},{shift[2]:.5f},'
                     f'{rec["px_supervised"]:.0f},{rec["grad_norm"]:.5f},'
                     f'{rec["clip_frac"]:.4f},{ep_t:.1f}\n')
        save_curves(history)

        _ck = {'epoch': epoch, 'run_id': RUN_ID, 'best_psnr': best_psnr,
               'loss_scale': loss_scale, 'optimizer': optimizer.state_dict(),
               'lora_state': lora.state_dict()}
        torch.save(_ck, _CKPT / f'lora_e{epoch:03d}.pt')
        if new_best:
            torch.save(_ck, _CKPT / 'lora_best.pt')
            print('  [CKPT] new best -> lora_best.pt', flush=True)

        if epoch % DIAG_EVERY == 0:
            ed = _DIAG / f'e{epoch:03d}'; ed.mkdir(exist_ok=True)
            for fi in DIAG_FRAMES:
                slat = _slat_from_cache(slat_cache, fi)
                mesh = colonly_forward(dec_model, lora, slat, with_grad=False)
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
                    (r, f'intersection e{epoch:03d}'),
                ]).save(ed / f'strip_f{fi:04d}.png')
                del r, mesh, c
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

    print(f'\n  METRICS ON v4\'s REGION (directly comparable):')
    print(f'    All:   PSNR={res_all["psnr_mean"]:.3f}+-{res_all["psnr_std"]:.3f}   (v4: 10.647)')
    print(f'    Held:  PSNR={res_held["psnr_mean"]:.3f}+-{res_held["psnr_std"]:.3f}   (v4: 10.664)')
    print(f'    Train: PSNR={res_train["psnr_mean"]:.3f}+-{res_train["psnr_std"]:.3f}   (v4: 10.645)')
    print(f'    SSIM={res_all["ssim_mean"]:.4f} (v4: 0.8671)   '
          f'LPIPS={res_all["lpips_mean"]:.4f} (v4: 0.1256)')
    print(f'\n  METRICS ON THE INTERSECTION (do NOT compare to v4):')
    print(f'    All:   PSNR={res_all["psnr_inter_mean"]:.3f}')
    print(f'\n  SUCCESS CRITERION  (15 frames):')
    print(f'    dR={final_shift[0]:+.4f}  dG={final_shift[1]:+.4f}  dB={final_shift[2]:+.4f}')
    print(f'    v4 was  +0.2457 / +0.2233 / +0.2380')
    print(f'    target  +0.1572 / +0.0853 / +0.0793')
    verdict = 'CONFIRMED' if final_shift[2] < 0.16 else 'NOT CONFIRMED'
    print(f'    -> white-cast hypothesis {verdict} '
          f'(dB {"fell below" if final_shift[2] < 0.16 else "stayed above"} 0.16)')

    json.dump({'run_id': RUN_ID, 'held': res_held, 'train': res_train, 'all': res_all,
               'final_shift': [float(x) for x in final_shift],
               'v4_shift': [0.2457, 0.2233, 0.2380],
               'intersection_prediction': [0.1572, 0.0853, 0.0793],
               'verdict': verdict},
              open(_OUT / 'final_eval.json', 'w'), indent=2)
    print(f'\n[DONE]  best held PSNR={best_psnr:.3f} dB   output={_OUT}', flush=True)


if __name__ == '__main__':
    main()
