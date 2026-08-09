"""
rung18_satpenalty.py
--------------------
rung16_e1_rembgmask.py with ONE change: a penalty on colour logits that enter
sigmoid saturation.  rung15 and rung16 are untouched and still run.

WHAT WAS MEASURED (visibility/logits/logits.json, 7 frames, 11.2M logits each)

  Vertex colour is sigmoid(raw), flexicubes.py:94.  The raw logits live in
  out_layer channels 53:101 (cube2mesh.py: sdf 0:8 | deform 8:32 | weights 32:53
  | color 53:101 = 8 corners x 6ch).

                 p50      p99      max     >3       >4
      frozen    -2.30     1.10     2.94   0.00%    0.00%
      adapted   -1.16     3.20     5.71   2.14%    0.16%

  FROZEN TRELLIS NEVER EXCEEDS 2.94.  Not once.  The adapter reaches 5.71 and
  puts ~0.16% of logits (about 18,000 per frame) past 4.

      sigmoid(2.9) = 0.948     sigmoid(4.0) = 0.982     sigmoid(5.7) = 0.997
      d/dx at 4.0  = 0.018     at 6.0 = 0.0025

  Past 4 the curve is flat: the loss loses its grip, which is why rung16's 8.85x
  stronger MSE raised PSNR but left the white patches.  Those ~18,000 logits ARE
  the white patches.

THE CHANGE

      L = MSE + W_LPIPS*LPIPS + LAMBDA * mean( relu(|raw[53:101]| - TAU)^2 )

  TAU = 4.0 is set by where sigmoid dies, NOT by this teapot -- any object may
  still reach colour 0.982 for free.  It is a property of the activation, so it
  transfers to any dataset.

WHY NOT THE OTHER TWO CANDIDATES
  * A CLAMP like rung13's clamp(-9, 8): the ceiling is 8, the adapter only
    reaches 5.71, so it would never fire.  And clamp has ZERO gradient outside
    its range -- it stops a value going further but never pulls one back.
  * A DRIFT penalty |raw_lora - raw_frozen|: frozen is too DARK (mean albedo
    0.176 vs GT 0.274), so pulling toward frozen fights the adapter's actual
    job.  Rejected for that reason.

WHAT THIS DOES NOT ADDRESS, STATED PLAINLY
  Blown-out pixels are worst at 45-90 degrees from the training view (5.20%) and
  BETTER at the back (2.01%) than at the sides.  This penalty is angle-agnostic;
  it does not explain or target that asymmetry.  If the patches survive with a
  large LAMBDA, the cause is geometric (grazing-angle surface being
  under-weighted in an image-space loss) and this is the wrong fix.

  LAMBDA = 0 reproduces rung16 exactly and is the control.

--------------------------------------------------------------------------
INHERITED FROM rung16 (unchanged):
rung15_v1_xattn_randt.py with ONE change: the mask the TRAINING loss is computed
over.

THE BUG IN EVERY RUNG SO FAR (5 through 15)

      gt_mask = (gt < 0.99).any(dim=0)

  is true wherever ANY channel is below 0.99.  The video background is off-white
  (~0.98) so it passes, and the mask covers 99.7% of the frame instead of the
  11.5% that is teapot.  Since

      mse = sum((render - gt)^2 * m) / (sum(m) * 3)

  the denominator is ~8.7x too large and MSE is silently shrunk by that factor.
  Measured on rung15's own epoch log, the nominal 10:1 MSE:LPIPS weighting ran at

      epoch  1   MSE 0.01161   0.1*LPIPS 0.00779   ->  1.49 : 1
      epoch 30   MSE 0.00333   0.1*LPIPS 0.00187   ->  1.79 : 1

  MSE is the only term that penalises ABSOLUTE brightness; LPIPS compares
  pretrained features and is near-blind to a uniform brightness shift.  So half
  the objective did not care how bright the teapot was.  rung15's albedo came out
  2.6x frozen's, with 2.48% of rendered pixels blown out against frozen's 0.43%.

THE CHANGE

      gt_mask = cached u2net foreground, alpha > 0.8*255

  This is the segmentation TRELLIS itself applies to its conditioning images
  (trellis_image_to_3d.py:100-105), so the loss definition carries to any video
  rather than depending on this one having a bright background.  A brightness
  rule (min(R,G,B) < 0.95) agrees with it at IoU 0.979 here and is far cheaper,
  but only because this backdrop is uniform; --train-mask brightness selects it.

  Expected effect: sum(m) 268,000 -> ~30,000, so MSE x8.7 and the balance moves
  from 1.8:1 to ~15:1.

TRAINING ONLY.  evaluate_frames still uses load_gt's leaky mask, deliberately:
  its LPIPS is computed over (render_mask | gt_mask), so changing gt_mask there
  would change what the reported LPIPS MEANS and rung15's 0.0215 would stop
  being comparable.  Held PSNR/SSIM/LPIPS in this run are therefore measured
  exactly as rung15 measured them.

Everything else is rung15 verbatim: cross-attention LoRA on all 24 blocks
(to_q, to_kv, to_out), rank 4, seed 6, 30 epochs, lr 1e-4, LOSS_SCALE 4096,
W_LPIPS 0.1, random-knot gradient, rung13 alignment, same held-out split.

--------------------------------------------------------------------------
INHERITED FROM rung15 (unchanged):
rung14_v1_crossattn_lora.py with ONE change: the timestep the gradient is taken
at.

THE BUG IN rung14-v1

  25 denoising steps.  rung14-v1 ran all 25 with the adapter active but took the
  gradient ONLY at the last one (T_PAIRS[-1], t = 0.1111 -> 0).  Steps 1-24 —
  88.9% of the ODE path — got no feedback at all.

  That is the worst possible place to train a CROSS-ATTENTION adapter.  Cross
  attention is how the image reaches the 3D model, and the image decides
  structure EARLY, at high t.  By t = 0.11 the latent is essentially formed and
  the model is refining detail.  So the adapter was fitted where its job barely
  matters, then deployed at all 25 steps including the early ones it never saw.
  Train/test mismatch, inherited from rung2 without re-examination.

  Result: 20.626 dB held.  That number is not evidence about cross-attention.
  It is evidence about training cross-attention at one late timestep.

THE FIX

  Sample k uniformly over the 25 schedule knots each training step:

      x_k   = denoise steps 0..k-1                       no_grad, LoRA ACTIVE
      v     = flow(x_k, t_k, cond)                       ONE eval, WITH grad
      x0hat = (1-s)*x_k - (s + (1-s)*t_k) * v            s = SIGMA_MIN = 1e-5
      loss  = compare(render(decode(normalize(x0hat))), frame)

  x0hat is FlowEulerSampler._v_to_xstart_eps (flow_euler.py:35) verbatim, not a
  re-derivation, so the rectified-flow parameterisation cannot drift from
  TRELLIS's.

  Uniform over KNOTS, not over t.  With rescale_t=3.0 the knots are heavily
  front-loaded (t_0..t_22 all sit above 0.29), so uniform-over-knots already
  concentrates where inference actually goes.  This is exactly what the
  TRELLIS.2 rung14 does (rung14_crossattn_lora.py:987) and it is the only part
  of this design that is already validated.

  INFERENCE IS UNCHANGED.  Eval, diagnostics and every gate still run the full
  25-step ODE.  Only where the gradient is taken moved.

WHAT THIS COSTS

  rung14-v1: 24 no-grad + 1 grad + 1 replay = 26 flow evals per frame.
  rung15   : E[k] = 12 no-grad + 1 grad + 1 replay = 14.
  About half the wall clock — ~250 s/epoch against rung14-v1's 517.

HONEST CAVEAT
  This optimises each timestep's ONE-STEP reconstruction independently.  It is
  not a lower-variance estimator of rung14-v1's objective; it is a DIFFERENT
  objective — the standard flow-matching one.  The case for it is empirical
  (TRELLIS.2 rung14 reached 20.86 dB by epoch 12 with it), not a proof that it
  bounds the true trajectory gradient.

Everything below this point is rung14-v1 verbatim except _train_step, the
timestep helpers, and the k-logging.

---------------------------------------------------------------------------
LoRA on the SLaT FLOW MODEL's CROSS-ATTENTION — i.e. BEFORE the decoder — on
TRELLIS v1, with rung13's mesh alignment applied.

WHAT IS NEW vs rung2/rung3 (which also hit cross-attention, in Aug 2026)

  rung2_placement.py and rung3_grid.py already put LoRA on
  slat_flow_model.blocks[i].cross_attn.  They are UNTOUCHED and still run.
  Three things were wrong or missing in them, and all three are fixed here:

    1. NO ALIGNMENT.  They trained against the misregistered render (11.2%
       wider, 16 px off, 7 deg rotated, silhouette IoU 0.760).  rung13 later
       showed that costs 1.37 dB on the decoder-side arm
       (rung5_colonly 21.281 -> rung13 22.646, alignment the only change).
       Here the solved transform is applied in render_mesh(), so cross-attention
       is finally measured on the same footing as rung13.

    2. NO to_out.  rung2 adapted to_q + to_kv only.  rung3 added to_out but ALSO
       fc1/fc2, so the two effects were confounded.  Default here is
       to_q + to_kv + to_out and nothing else -- the exact target set the
       TRELLIS.2 rung14 uses, so the two are directly comparable.
       --targets qkv reproduces rung2's set.

    3. NO GRADIENT INSTRUMENTATION.  Neither logged per-target or per-block
       gradient norms, so "did it train" was never separable from "did it help".

  Default is all 24 blocks.  rung2/rung3 swept 8-block windows
  (early 0-7 / mid 8-15 / late 16-23); --blocks reproduces those.

GEOMETRY IS NOT FROZEN HERE.  READ THIS BEFORE USING THE NUMBERS.

  rung13 could guarantee frozen geometry because it adapted dec_model.out_layer,
  whose 101 output channels split cleanly:

      [0:8] sdf   [8:32] deform   [32:53] FlexiCubes   [53:101] colour

  so geometry channels could be taken from a frozen pass and .detach()ed, and
  only [53:101] came from the LoRA pass.  That splice is what made "geometry
  provably unchanged, 0/150" true.

  Cross-attention is upstream of the SLaT latent, and trellis_image_to_3d.py:210
  feeds that ONE latent to slat_decoder_mesh(slat), which emits geometry AND
  colour together.  There are no separate channels to splice upstream, so there
  is no way to adapt v1 cross-attention and hold geometry fixed.  Geometry WILL
  move.

  This script therefore does not claim otherwise.  It MEASURES the drift every
  epoch at frame 75 -- vertex-count delta, silhouette IoU against the frozen
  render, and silhouette IoU against the tight GT mask -- and writes them to
  logs/geometry_drift.csv.  If the drift is small the result is usable; if it is
  large it is the "ghost"/second-teapot failure that the decoder-block
  placements showed (visibility/ghost_measurement/ghost_measurement.json:
  blk 8-11 went 215,606 -> 280,756 verts, n_components 1.0 -> 2.6).
  That is an empirical question and this run answers it.

GRADIENT PATH (truncated backprop -- rung2's, kept because it is proven)

  25 flow steps.  Steps 1..24 run under no_grad WITH the LoRA active (the
  adapter changes the trajectory, so it must be on).  Only step 25 carries
  gradient.  The backward is two-stage to keep the flow graph and the decoder
  graph from co-existing in memory:

      x_prefix = denoise 1..24                                   (no_grad)
      slat_ng  = denoise step 25                                 (no_grad)
      leaf     = slat_ng.feats.detach().requires_grad_(True)
      flow_model.cpu()                       <- frees ~4 GB
      decode(leaf) -> render -> loss -> backward   => leaf.grad
      flow_model.cuda()
      slat_r   = denoise step 25                                 (WITH grad)
      autograd.backward(slat_r.feats, leaf.grad)  => LoRA grads

  Costs one extra step-25 evaluation per frame; that is the price of not
  holding both graphs.  Attention is TRELLIS's own fused kernel, not rung2's
  chunked float32 reimplementation -- see _xattn_fwd for why that mattered.

COMPARABILITY
  seed 6, rank 4, 30 epochs, lr 1e-4, LOSS_SCALE 4096, W_LPIPS 0.1,
  held-out [5,15,...,145], same GT frames, same union loss, same leaky gt_mask
  as rung13 -- so held-out PSNR here is directly comparable to rung13's 22.646.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  python experiments/lora_experiments/rung14_v1_crossattn_lora.py \
    --rank 4 --epochs 30 --seed 6 --blocks all --targets qkvo
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
_ROOT  = _HERE.parent.parent
_PIPE  = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_RUNS  = _HERE / 'runs'
_RUNS.mkdir(parents=True, exist_ok=True)

_ap_ = _ap.ArgumentParser()
_ap_.add_argument('--rank',    type=int,   default=4)
_ap_.add_argument('--seed',    type=int,   default=6)
_ap_.add_argument('--epochs',  type=int,   default=30)
_ap_.add_argument('--blocks',  default='all', choices=['all', 'early', 'mid', 'late'],
                  help='which cross-attn blocks get the adapter. '
                       'early=0-7 mid=8-15 late=16-23 reproduce rung2/rung3.')
_ap_.add_argument('--targets', default='qkvo', choices=['qkv', 'qkvo'],
                  help='qkv = rung2 set (to_q,to_kv).  qkvo = + to_out, the '
                       'TRELLIS.2 rung14 set.  MLP is never touched here — '
                       'that was rung3 and it lost by 1.6 dB.')
_ap_.add_argument('--alignment', default=None, type=Path,
                  help='alignment.json from solve_mesh_alignment.py')
_ap_.add_argument('--no-align', action='store_true',
                  help='disable the transform. Only for reproducing rung2.')
_ap_.add_argument('--lam', type=float, default=3.0,
                  help='weight on the saturation penalty. 0 reproduces rung16.')
_ap_.add_argument('--tau', type=float, default=4.0,
                  help='logit magnitude past which sigmoid is flat. sigmoid(4)'
                       '=0.982, gradient 0.018; by 6 it is 0.0025 and dead.')
_ap_.add_argument('--train-mask', default='rembg',
                  choices=['rembg', 'brightness', 'leaky'],
                  help="mask the TRAINING loss is averaged over. rembg = cached "
                       "u2net (what TRELLIS uses). brightness = min(RGB)<0.95, "
                       "IoU 0.979 with rembg here but background-dependent. "
                       "leaky = rung15's (gt<0.99).any(), i.e. the control.")
_ap_.add_argument('--tsample', default='uniform', choices=['uniform', 'late', 'last'],
                  help="which schedule knot the gradient is taken at each step. "
                       "uniform = k ~ U{0..24}, what TRELLIS.2 rung14 does. "
                       "late = k ~ U{12..24}, if high-t renders prove too noisy "
                       "to supervise. last = k=24 always, i.e. reproduce "
                       "rung14-v1 exactly — the control for this experiment.")
_ap_.add_argument('--smoke',   action='store_true')
_ap_.add_argument('--gates-only', action='store_true',
                  help='run every gate incl. one real training step, print peak '
                       'GPU, then exit. ~10 min. Use before burning a 4 h slot.')
_ap_.add_argument('--offload', default='auto', choices=['auto', 'always', 'never'],
                  help="park the 4 GB flow model on the CPU during decode+render. "
                       "rung2 needed this on an 11.7 GB card. On a 48 GB A40 it is "
                       "pure PCIe overhead — 8 GB of traffic per frame, 135 frames "
                       "per epoch. auto = offload only if the GPU has < 32 GB.")
_ap_.add_argument('--diag-every', type=int, default=5)
args = _ap_.parse_args()

# ── rung13's solved mesh alignment ────────────────────────────────────────────
USE_ALIGN = not args.no_align
_ALIGN_PATH = args.alignment or (_HERE / 'visibility' / 'alignment' / 'alignment.json')
if USE_ALIGN and not _ALIGN_PATH.exists():
    raise FileNotFoundError(
        f'alignment not found: {_ALIGN_PATH}\n'
        f'run visibility/solve_mesh_alignment.py first, or pass --no-align.')
if USE_ALIGN:
    _ALIGN = json.load(open(_ALIGN_PATH))
    ALIGN_S      = float(_ALIGN['scale'])
    ALIGN_ROTVEC = [float(x) for x in _ALIGN['rotvec']]
    ALIGN_T       = [float(x) for x in _ALIGN['translation']]
    ALIGN_CENTRE = [float(x) for x in _ALIGN['centre']]
else:
    ALIGN_S, ALIGN_ROTVEC = 1.0, [0.0, 0.0, 0.0]
    ALIGN_T, ALIGN_CENTRE = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

_BLOCK_SETS = {
    'all':   list(range(24)),
    'early': list(range(0, 8)),
    'mid':   list(range(8, 16)),
    'late':  list(range(16, 24)),
}
ACTIVE_BLOCKS = _BLOCK_SETS[args.blocks]
TARGETS = ('to_q', 'to_kv') if args.targets == 'qkv' else ('to_q', 'to_kv', 'to_out')

# ── constants (identical to rung13 so the numbers are comparable) ─────────────
GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED   = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES     = 150
STRUCT_SEED  = 42
STEPS        = 25
RESCALE_T    = 3.0
LR           = 1e-4
LOSS_SCALE0  = 4096.0
GRAD_CLIP    = 1.0
W_LPIPS      = 0.1
N_XATTN      = 24          # asserted at runtime

HELD_OUT = list(range(5, 151, 10))
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

_CFG = dict(
    variant='v1_crossattn_lora_randt_rembgmask_satpen',
    lam=args.lam, tau=args.tau,
    train_mask=args.train_mask,
    tsample=args.tsample,
    targets=list(TARGETS), active_blocks=ACTIVE_BLOCKS,
    aligned=USE_ALIGN,
    align_scale=ALIGN_S, align_rotvec=ALIGN_ROTVEC,
    align_t=ALIGN_T, align_centre=ALIGN_CENTRE,
    rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    steps=STEPS, rescale_t=RESCALE_T, held_out=HELD_OUT,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = (f'rung18sp_lam{args.lam}_{args.train_mask}_{args.tsample}_{args.blocks}'
          f'_{args.targets}_r{args.rank}_s{args.seed}_{RUN_ID}')
_OUT   = _RUNS / _LABEL
_CKPT  = _OUT / 'lora_ckpts'
_DIAG  = _OUT / 'diag_renders'
_LOGD  = _OUT / 'logs'
for _d in (_OUT, _CKPT, _DIAG, _LOGD):
    _d.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(_OUT / 'train.log')
sys.stderr = sys.stdout


# ── nvdiffrast arch guard (verbatim from rung13) ──────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
import math, time, gc, random
from contextlib import contextmanager
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
from trellis.modules.sparse.attention import sparse_scaled_dot_product_attention
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM  = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# Resolved in main() once the GPU is known. rung2 hard-coded the offload because
# it ran on an 11.7 GB card; on a 48 GB one it costs ~8 GB of PCIe traffic per
# frame and buys nothing.
OFFLOAD = True


@contextmanager
def parked(flow_model):
    """Move the flow model off the GPU for the decode+render+backward window."""
    if OFFLOAD:
        flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    try:
        yield
    finally:
        if OFFLOAD:
            flow_model.to(DEVICE)


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """B is zero-init, so at step 0 the model is bit-identical to frozen."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class XAttnLoRABundle(nn.Module):
    """
    One bundle per adapted cross-attn block.  Shapes are read off the REAL
    nn.Linear modules rather than hardcoded, so a model with different widths
    cannot silently produce a mis-shaped adapter.
    """
    def __init__(self, ca, rank: int, targets):
        super().__init__()
        self.targets = tuple(targets)
        for name in self.targets:
            lin = getattr(ca, name)
            setattr(self, f'lora_{name}',
                    LoRALayer(lin.in_features, lin.out_features, rank))

    def layers(self):
        return [(n, getattr(self, f'lora_{n}')) for n in self.targets]

    def get(self, name):
        """None if this bundle does not adapt `name`. Called in the inner loop,
        so it must not build a dict every time."""
        return getattr(self, f'lora_{name}', None) if name in self.targets else None


class XAttnLoRARegistry(nn.Module):
    def __init__(self, flow_model, active_blocks, rank: int, targets):
        super().__init__()
        self.active  = set(active_blocks)
        self.targets = tuple(targets)
        mods = {}
        idx = 0
        for blk in flow_model.blocks:
            if not hasattr(blk, 'cross_attn'):
                continue
            if idx in self.active:
                mods[str(idx)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            idx += 1
        self.n_xattn = idx
        self.blocks  = nn.ModuleDict(mods)

    def get(self, block_idx: int):
        key = str(block_idx)
        return self.blocks[key] if key in self.blocks else None


# ── the patched cross-attention forward ───────────────────────────────────────

def _xattn_fwd(module, x, context, lb):
    """
    Replacement for SparseMultiHeadAttention.forward, cross-attention branch.

    THIS MIRRORS modules.py:126-139 LINE FOR LINE and calls TRELLIS's OWN
    sparse_scaled_dot_product_attention.  The only additions are the three LoRA
    deltas.  With B=0 the result is bit-identical to the unpatched model, which
    GATE-identity asserts.

    WHY NOT rung2's HAND-ROLLED ATTENTION.  rung2's _placement_fwd (and rung3's)
    replaced the fused attention with a chunked einsum+softmax in float32.  The
    model runs fp16 (slat_flow_img_dit_L_64l8p2_fp16.json: use_fp16 true), so
    that swaps the numerics.  Measured here, at B=0, on the SLaT after 25 steps:

        rung2's chunked fp32 attention   1.801e-02
        this (model's own attention)     6.800e-03
        the model's own run-to-run noise 1.065e-02

    So rung2's version was ~1.7x the noise floor -- elevated, but NOT the large
    confound it first looked like, and it would have passed a 3x-noise gate.
    The honest reason to use TRELLIS's own attention is not that rung2 was
    broken; it is that this removes the question entirely and is also cheaper --
    xformers' kernel never materialises the N x M matrix, which is the very
    thing rung2's chunking existed to avoid.

    lb is None -> inactive block.  Still runs with autograd enabled so gradient
                  can TRAVEL THROUGH it to an adapted block upstream (with
                  --blocks early, blocks 8-23 sit between the loss and the
                  adapters; cutting their graph would bias every gradient).
    """
    # q: SparseTensor [N, C] -> [N, H, hd]
    q_sp = module._linear(module.to_q, x)
    if lb is not None:
        l_q = lb.get('to_q')
        if l_q is not None:
            q_sp = q_sp.replace(q_sp.feats + l_q(x.feats).to(q_sp.feats.dtype))
    q = module._reshape_chs(q_sp, (module.num_heads, -1))

    # kv: [1, N_ctx, 2C] -> [1, N_ctx, 2, H, hd]
    kv_t = module._linear(module.to_kv, context)
    if lb is not None:
        l_kv = lb.get('to_kv')
        if l_kv is not None:
            kv_t = kv_t + l_kv(context).to(kv_t.dtype)
    kv = module._fused_pre(kv_t, num_fused=2)

    # qk_rms_norm_cross defaults False and the L-64l8p2 config does not set it,
    # so this is dead for our checkpoint. Kept so a config that DOES enable it
    # fails loudly instead of silently skipping the norm.
    if module.qk_rms_norm:
        raise NotImplementedError(
            'cross-attn qk_rms_norm is enabled on this checkpoint; the patched '
            'forward does not implement it and would diverge from frozen.')

    h = sparse_scaled_dot_product_attention(q, kv)
    h = module._reshape_chs(h, (-1,))
    out = module._linear(module.to_out, h)
    if lb is not None:
        l_o = lb.get('to_out')
        if l_o is not None:
            out = out.replace(out.feats + l_o(h.feats).to(out.feats.dtype))
    return out


@contextmanager
def xattn_lora_ctx(flow_model, registry: XAttnLoRARegistry):
    """Patch every cross-attn block; inactive ones run frozen."""
    saved  = {}
    ca_idx = 0
    for blk in flow_model.blocks:
        if not hasattr(blk, 'cross_attn'):
            continue
        ca            = blk.cross_attn
        saved[ca_idx] = ca.forward
        lb            = registry.get(ca_idx)

        def _make(mod, lb_):
            def _fwd(x, context=None):
                return _xattn_fwd(mod, x, context, lb_)
            return _fwd

        ca.forward = _make(ca, lb)
        ca_idx += 1

    assert ca_idx == N_XATTN, f'expected {N_XATTN} cross-attn blocks, found {ca_idx}'
    try:
        yield
    finally:
        ca_idx = 0
        for blk in flow_model.blocks:
            if hasattr(blk, 'cross_attn') and ca_idx in saved:
                blk.cross_attn.forward = saved[ca_idx]
                ca_idx += 1


# ── denoising ─────────────────────────────────────────────────────────────────

SIGMA_MIN = 1e-5     # pipeline.json: slat_sampler.args.sigma_min

# The set of knots the gradient may be taken at. Uniform over KNOTS, not over t:
# with rescale_t=3.0 the knots are front-loaded (t_0..t_22 all above 0.29), so
# this already concentrates where inference spends its time.
_TS_POOL = {
    'uniform': list(range(STEPS)),          # 0..24  — TRELLIS.2 rung14's choice
    'late':    list(range(STEPS // 2, STEPS)),   # 12..24
    'last':    [STEPS - 1],                 # 24 only == rung14-v1, the control
}[args.tsample]

_RNG = np.random.default_rng(args.seed)

# The knot most recently drawn. _train_step returns None on a skipped step, so
# the caller would otherwise have no way to know which knot failed.
_LAST_K = [ -1 ]


def sample_k():
    return int(_TS_POOL[int(_RNG.integers(0, len(_TS_POOL)))])


def denoise_prefix_to_k(flow_model, noise_sp, cond_gl, registry, k):
    """
    Steps 0..k-1 with the LoRA ACTIVE but no gradient retained.

    The adapter MUST be active here even though nothing is learned from it: it
    changes the trajectory, so x_k has to be the state the adapted model would
    actually reach. Running the prefix frozen would hand the graded step an
    input the deployed model never produces.

    k=0 returns the pure-noise input unchanged, which is correct — t_0 = 1.0.
    """
    x = noise_sp
    with torch.no_grad():
        for j in range(k):
            t, t_prev = T_PAIRS[j]
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with xattn_lora_ctx(flow_model, registry):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def flow_eval_at(flow_model, x, t, cond_gl, registry, require_grad: bool):
    """One velocity evaluation at knot t, LoRA active."""
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx = _nullctx() if require_grad else torch.no_grad()
    with ctx:
        with xattn_lora_ctx(flow_model, registry):
            return flow_model(x, t_ten, cond_gl)


def pred_to_xstart(x_t, t, v):
    """
    FlowEulerSampler._v_to_xstart_eps, flow_euler.py:35, verbatim:

        x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v

    Copied rather than re-derived so the rectified-flow parameterisation cannot
    silently diverge from the one TRELLIS samples with. No division by (1 - t),
    so t = 1.0 (knot 0) is not a singularity.

    AT THE FINAL KNOT THIS *IS* rung14-v1's UPDATE. k=24 runs t=0.1111 -> 0, so
    rung14-v1's Euler step was  x - 0.111111*v  while this gives
    0.999990*x - 0.111120*v — the coefficients agree to 8.9e-06. That is what
    makes `--tsample last` a genuine control for this experiment rather than a
    second change smuggled in alongside the first.
    """
    s = SIGMA_MIN
    return x_t.replace((1.0 - s) * x_t.feats - (s + (1.0 - s) * t) * v.feats)


def full_denoise_nograd(flow_model, noise_feats, coords, cond_gl, registry=None):
    """All 25 steps, no grad. registry=None -> the frozen model."""
    ns = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            if registry is None:
                v = flow_model(ns, t_ten, cond_gl)
            else:
                with xattn_lora_ctx(flow_model, registry):
                    v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    return normalize_slat(ns)


# ── helpers (identical to rung13) ─────────────────────────────────────────────

def encode_frame(dino_model, frame_idx):
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt(frame_idx):
    img     = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img     = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt      = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = (gt < 0.99).any(dim=0)
    return gt, gt_mask


_MASK_NPZ = _HERE / 'gt_masks_rembg.npz'
TRAIN_MASKS = None          # [N_FRAMES, RENDER_RES, RENDER_RES] bool, on DEVICE


_RAW = {}          # out_layer's output for the current graded decode


def sat_penalty(raw, tau):
    """
    mean( relu(|raw| - tau)^2 ) over the COLOUR logits only, channels 53:101.

    Zero while the logit stays inside +-tau, so it costs nothing for the
    brightening the adapter legitimately needs (measured: median -2.30 -> -1.16,
    p99 1.10 -> 3.20, all inside 4). It bites only on the 0.16% that crossed into
    the flat part of sigmoid. Unlike a clamp it has a real gradient there, so it
    pulls values back rather than merely stopping them.
    """
    c = raw[:, 53:101]
    return (torch.relu(c.abs() - tau) ** 2).mean()


def load_train_masks():
    """
    The mask the TRAINING loss averages over. Returns a [150,H,W] bool tensor
    indexed by frame-1.

    Kept separate from load_gt() on purpose: load_gt still returns the leaky
    mask, which evaluate_frames uses, so the reported LPIPS keeps rung15's
    definition and the two runs stay comparable.
    """
    if args.train_mask == 'leaky':
        m = torch.stack([load_gt(i)[1] for i in range(1, N_FRAMES + 1)])
        print(f'[MASK] leaky (gt<0.99).any() -- rung15 control')
    elif args.train_mask == 'brightness':
        ms = []
        for i in range(1, N_FRAMES + 1):
            img = Image.open(GT_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB') \
                       .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
            a = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1)
            ms.append((a.min(dim=0).values < 0.95))
        m = torch.stack(ms).to(DEVICE)
        print(f'[MASK] brightness min(RGB)<0.95')
    else:
        assert _MASK_NPZ.exists(), (
            f'{_MASK_NPZ} missing -- run precompute_rembg_masks.py first. '
            f'u2net is CPU-only here (~53 min for 150 frames), which is why it '
            f'is cached rather than computed per run.')
        z = np.load(_MASK_NPZ)
        meta = json.loads(str(z['meta']))
        assert meta['res'] == RENDER_RES, \
            f"cached masks are {meta['res']}px, RENDER_RES is {RENDER_RES}"
        assert list(z['frames']) == list(range(1, N_FRAMES + 1)), \
            'cached mask frame list does not match 1..150'
        m = torch.from_numpy(z['masks']).to(DEVICE)
        print(f"[MASK] rembg u2net, alpha>{meta['alpha_thresh']}*255  "
              f"(cached {_MASK_NPZ.name})")
    assert m.shape == (N_FRAMES, RENDER_RES, RENDER_RES), f'bad shape {tuple(m.shape)}'
    frac = m.float().mean(dim=(1, 2))
    print(f'  foreground {100*float(frac.mean()):.2f}% +- {100*float(frac.std()):.2f}%'
          f'   (rung15 trained on 99.7%)', flush=True)
    return m


def load_render_mask():
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)


def tight_gt_mask(frame_idx=75):
    """The real teapot silhouette. load_gt's (gt<0.99) leaks onto the off-white
    background and covers ~99.7% of the frame — useless for IoU."""
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB') \
               .resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    a   = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    return (a.min(dim=0).values < 0.95)


def masked_psnr(pred, gt, mask):
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred, gt):
    p = pred.permute(1, 2, 0).cpu().numpy()
    g = gt.permute(1, 2, 0).cpu().numpy()
    return float(_ssim(p, g, data_range=1.0, channel_axis=2))


def masked_loss(rendered, gt, render_mask, gt_mask, lpips_fn):
    m   = (render_mask | gt_mask).float()
    mse = ((rendered - gt) ** 2 * m).sum() / (m.sum() * 3 + 1e-8)
    r   = rendered * m + (1 - m)
    g   = gt       * m + (1 - m)
    lp  = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).mean()
    return mse, lp, mse + W_LPIPS * lp


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def _rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k  = np.asarray(rv, dtype=np.float64) / th
    K  = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


_AR = torch.tensor(_rodrigues(ALIGN_ROTVEC), dtype=torch.float32, device=DEVICE)
_AC = torch.tensor(ALIGN_CENTRE, dtype=torch.float32, device=DEVICE)
_AT = torch.tensor(ALIGN_T,      dtype=torch.float32, device=DEVICE)


def align_vertices(v):
    return ALIGN_S * ((v - _AC) @ _AR.T) + _AC + _AT


def render_mesh(mesh, renderer, aligned=None):
    """Single choke point: training, eval, diagnostics and gates all render here."""
    if aligned is None:
        aligned = USE_ALIGN
    ext, intr = EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    _saved = mesh.vertices
    if aligned:
        mesh.vertices = align_vertices(mesh.vertices)
    res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    mesh.vertices = _saved
    mask = res['mask'].unsqueeze(0)
    return res['color'] * mask + (1.0 - mask), mask


# ── gradient instrumentation ──────────────────────────────────────────────────

def b_grad_stats(registry):
    """
    Reported on B, NOT A.  B is zero-init, so if B's gradient is zero then A's is
    zero by the chain rule and nothing can ever move.  Checking A would mislead.
    """
    per_target, per_block, all_n = {}, {}, []
    for bidx, blk in registry.blocks.items():
        bl = []
        for name, lyr in blk.layers():
            g = lyr.B.grad
            n = float(g.norm().item()) if g is not None else 0.0
            per_target.setdefault(name, []).append(n)
            bl.append(n); all_n.append(n)
        per_block[int(bidx)] = float(np.mean(bl)) if bl else 0.0
    n_zero = sum(1 for n in all_n if n == 0.0)
    return {
        'per_target': {k: float(np.mean(v)) for k, v in per_target.items()},
        'per_block':  per_block,
        'n_total':    len(all_n),
        'n_zero':     n_zero,
        'min':  float(np.min(all_n))    if all_n else 0.0,
        'med':  float(np.median(all_n)) if all_n else 0.0,
        'max':  float(np.max(all_n))    if all_n else 0.0,
        'ratio': float(np.max(all_n) / max(np.min(all_n), 1e-30)) if all_n else 0.0,
    }


def b_norms(registry):
    return [float(lyr.B.float().norm().item())
            for blk in registry.blocks.values() for _, lyr in blk.layers()]


# ── geometry drift: the measurement this run exists to produce ────────────────

def measure_geometry_drift(flow_model, dec_model, registry, renderer,
                           noise_feats, coords, cond_75, frozen_ref, gm_tight):
    """
    Cross-attention cannot be geometry-frozen in v1.  Quantify how far it moved.

    frozen_ref: dict from _frozen_reference() — vertex count and silhouette of
                the unadapted model at frame 75.
    """
    slat = full_denoise_nograd(flow_model, noise_feats, coords, cond_75, registry)
    with torch.no_grad():
        mesh = dec_model(slat)[0]
    n_v = int(mesh.vertices.shape[0])
    _, m = render_mesh(mesh, renderer)
    sil = (m.squeeze(0) > 0.5)

    inter = int((sil & frozen_ref['sil']).sum())
    union = int((sil | frozen_ref['sil']).sum())
    iou_frozen = inter / max(union, 1)
    iou_gt = int((sil & gm_tight).sum()) / max(int((sil | gm_tight).sum()), 1)

    d_pct = 100.0 * (n_v - frozen_ref['verts']) / max(frozen_ref['verts'], 1)
    out = {
        'verts': n_v,
        'verts_frozen': frozen_ref['verts'],
        'verts_delta_pct': d_pct,
        'area_px': int(sil.sum()),
        'area_frozen_px': frozen_ref['area'],
        'iou_vs_frozen': iou_frozen,
        'iou_vs_gt': iou_gt,
        'iou_frozen_vs_gt': frozen_ref['iou_gt'],
        # The floor, carried alongside so no reader has to go looking for it.
        'verts_noise_pct': frozen_ref['verts_noise_pct'],
        'iou_noise': frozen_ref['iou_noise'],
        # Is the movement real, or is it the pipeline's own jitter?
        'verts_above_noise': abs(d_pct) > max(frozen_ref['verts_noise_pct'], 1e-9),
        'iou_above_noise':   iou_frozen < frozen_ref['iou_noise'],
    }
    del slat, mesh, m, sil
    gc.collect(); torch.cuda.empty_cache()
    return out


def _frozen_reference(flow_model, dec_model, renderer, noise_feats, coords,
                      cond_75, gm_tight):
    """
    The frozen mesh at frame 75, PLUS the noise floor of that measurement.

    The pipeline is not bit-reproducible (see GATE-identity), so the frozen mesh
    itself moves a little between identical runs. Without knowing by how much,
    "the adapter changed geometry by 12 vertices" is unreadable. So run it twice
    and record the disagreement; measure_geometry_drift then reports the
    adapter's effect against that floor.
    """
    def _once():
        slat = full_denoise_nograd(flow_model, noise_feats, coords, cond_75, None)
        with torch.no_grad():
            mesh = dec_model(slat)[0]
        n_v = int(mesh.vertices.shape[0])
        _, m = render_mesh(mesh, renderer)
        sil = (m.squeeze(0) > 0.5).clone()
        del slat, mesh, m
        gc.collect(); torch.cuda.empty_cache()
        return n_v, sil

    v1, s1 = _once()
    v2, s2 = _once()
    iou_noise = int((s1 & s2).sum()) / max(int((s1 | s2).sum()), 1)

    ref = {
        'verts': v1, 'sil': s1, 'area': int(s1.sum()),
        'iou_gt': int((s1 & gm_tight).sum()) / max(int((s1 | gm_tight).sum()), 1),
        # noise floor: what "no change at all" actually looks like
        'verts_noise': abs(v2 - v1),
        'verts_noise_pct': 100.0 * abs(v2 - v1) / max(v1, 1),
        'iou_noise': iou_noise,
    }
    del s2
    gc.collect(); torch.cuda.empty_cache()
    return ref


# ── diagnostics ───────────────────────────────────────────────────────────────

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


def save_curves(history):
    if len(history) < 2:
        return
    ep   = [r['epoch']       for r in history]
    tot  = [r['loss_total']  for r in history]
    psnr = [r['held_psnr']   for r in history]
    bn   = [r['B_norm_mean'] for r in history]
    iou  = [r.get('geom_iou_vs_frozen', np.nan) for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'rung14-v1 cross-attn — {_LABEL}', fontsize=11)
    axes[0].plot(ep, tot,  '^-', color='#c678dd', lw=2, ms=4)
    axes[0].set_title('Train loss'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, psnr, 'o-', color='#98c379', lw=2, ms=4)
    axes[1].axhline(22.646, ls='--', c='#888', lw=1)
    axes[1].set_title('Held-out PSNR (dB)  — dashed = rung13'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, bn, 'o-', color='#61afef', lw=2, ms=4)
    axes[2].set_title('Mean ||B||'); axes[2].grid(True, alpha=0.3)
    axes[3].plot(ep, iou, 'o-', color='#e06c75', lw=2, ms=4)
    axes[3].axhline(1.0, ls='--', c='#888', lw=1)
    axes[3].set_title('Geometry IoU vs frozen\n(1.0 = unchanged)'); axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_frames(flow_model, dec_model, registry, renderer, render_mask,
                    lpips_fn, noise_feats, coords, raw_tokens, frame_list):
    """
    Two phases, so that when offloading is on the flow model crosses the PCIe
    bus twice per eval rather than twice per frame:
      phase 1 — denoise every frame, park each SLaT on the CPU
      phase 2 — decode + render every frame
    """
    slats = {}
    for fi in frame_list:
        cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        s    = full_denoise_nograd(flow_model, noise_feats, coords, cond, registry)
        slats[fi] = (s.feats.cpu(), s.coords.cpu())
        del s, cond
        gc.collect(); torch.cuda.empty_cache()

    per_frame = []
    with parked(flow_model):
        for fi in frame_list:
            f, c = slats[fi]
            slat = sp.SparseTensor(feats=f.to(DEVICE), coords=c.to(DEVICE))
            with torch.no_grad():
                mesh = dec_model(slat)[0]
            color, _ = render_mesh(mesh, renderer)
            render = color.detach().clamp(0, 1)
            gt, gt_mask = load_gt(fi)
            psnr = masked_psnr(render, gt, render_mask)
            ssim = compute_ssim(render, gt)
            m = (render_mask | gt_mask).float()
            r = render * m + (1 - m); g = gt * m + (1 - m)
            with torch.no_grad():
                lp = lpips_fn(r.unsqueeze(0) * 2 - 1, g.unsqueeze(0) * 2 - 1).item()
            per_frame.append({'frame': fi, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
            del render, gt, gt_mask, mesh, color, slat
            gc.collect(); torch.cuda.empty_cache()

    psnrs = [r['psnr'] for r in per_frame]
    ssims = [r['ssim'] for r in per_frame]
    lps   = [r['lpips'] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'lpips_mean': float(np.mean(lps)),  'lpips_std': float(np.std(lps)),
        'per_frame': per_frame,
    }


def save_diagnostics(flow_model, dec_model, registry, renderer, epoch,
                     noise_feats, coords, raw_tokens, baseline_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    for fi in DIAG_FRAMES:
        cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat = full_denoise_nograd(flow_model, noise_feats, coords, cond, registry)
        with torch.no_grad():
            mesh = dec_model(slat)[0]
        color, _ = render_mesh(mesh, renderer)
        render = color.detach().clamp(0, 1)

        if fi not in baseline_cache:
            slat_b = full_denoise_nograd(flow_model, noise_feats, coords, cond, None)
            with torch.no_grad():
                mesh_b = dec_model(slat_b)[0]
            cb, _ = render_mesh(mesh_b, renderer)
            baseline_cache[fi] = cb.detach().clamp(0, 1)
            del slat_b, mesh_b, cb

        strip = make_strip([
            (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
            (baseline_cache[fi],                    'frozen TRELLIS'),
            (render,                                f'xattn-LoRA e{epoch:03d}'),
        ])
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'render_f{fi:04d}.png')
        del render, mesh, color, slat, cond
        gc.collect(); torch.cuda.empty_cache()


def find_latest_ckpt():
    ckpts = sorted(_CKPT.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global OFFLOAD
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    _gp    = torch.cuda.get_device_properties(0)
    _gb    = _gp.total_memory / 1e9
    OFFLOAD = (_gb < 32.0) if args.offload == 'auto' else (args.offload == 'always')

    print('=' * 72)
    print('rung18 — rung16 + saturation penalty on out_layer colour logits')
    print(f'  lam / tau     : {args.lam} / {args.tau}   (frozen never exceeds 2.94; adapted reached 5.71)')
    print(f'  train_mask    : {args.train_mask}   (eval keeps rung15\'s leaky mask)')
    print(f'  tsample       : {args.tsample}  -> k in '
          f'{{{_TS_POOL[0]}..{_TS_POOL[-1]}}} ({len(_TS_POOL)} knots), '
          f't in [{T_PAIRS[_TS_POOL[-1]][0]:.4f}, {T_PAIRS[_TS_POOL[0]][0]:.4f}]')
    print(f'  (rung14-v1 took the gradient only at k=24, t=0.1111)')
    print(f'  blocks        : {args.blocks} -> {ACTIVE_BLOCKS}')
    print(f'  targets       : {TARGETS}')
    print(f'  aligned       : {USE_ALIGN}')
    print(f'  rank / seed   : {args.rank} / {args.seed}')
    print(f'  epochs        : {EPOCHS}')
    print(f'  run_id        : {RUN_ID}')
    print(f'  output        : {_OUT}')
    print(f'  GPU           : {_gp.name}  {_gb:.0f} GB')
    print(f'  offload       : {OFFLOAD}  (--offload {args.offload})')
    print('  NOTE: geometry is NOT frozen in this rung — it is MEASURED.')
    print('=' * 72)

    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL},
              open(_OUT / 'config.json', 'w'), indent=2)
    assert len(set(HELD_OUT) & set(TRAIN)) == 0

    print(f'\n[LOAD] {PRETRAINED}')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']

    for p in flow_model.parameters(): p.requires_grad_(False)
    for p in dec_model.parameters():  p.requires_grad_(False)
    n_frozen = sum(p.numel() for p in flow_model.parameters()) + \
               sum(p.numel() for p in dec_model.parameters())
    print(f'[FREEZE] {n_frozen:,} TRELLIS params frozen')

    # ── structure ─────────────────────────────────────────────────────────────
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

    # ── LoRA registry ─────────────────────────────────────────────────────────
    registry = XAttnLoRARegistry(flow_model, ACTIVE_BLOCKS, args.rank, TARGETS).to(DEVICE)
    assert registry.n_xattn == N_XATTN, \
        f'found {registry.n_xattn} cross-attn blocks, expected {N_XATTN}'
    n_params = sum(p.numel() for p in registry.parameters())

    # GATE 0 — param count derived from the REAL Linear shapes, not hardcoded.
    exp = 0
    _idx = 0
    for blk in flow_model.blocks:
        if not hasattr(blk, 'cross_attn'):
            continue
        if _idx in set(ACTIVE_BLOCKS):
            for nme in TARGETS:
                lin = getattr(blk.cross_attn, nme)
                exp += args.rank * (lin.in_features + lin.out_features)
        _idx += 1
    print(f'\n[LORA] blocks={len(ACTIVE_BLOCKS)}  targets={TARGETS}  '
          f'rank={args.rank}  params={n_params:,}')
    assert n_params == exp, f'GATE 0 FAILED: {n_params:,} != {exp:,}'
    print(f'[GATE 0] PASSED ({n_params:,} params)\n')

    # ── fixed noise + DINO cache ──────────────────────────────────────────────
    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    print(f'[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    # NOTE: no SLaT cache. rung13 could cache because its adapter lived
    # downstream of the flow. Here the adapter CHANGES the flow, so every step
    # must re-denoise. That is the cost of adapting cross-attention.

    print('\n[LPIPS] Loading AlexNet...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)

    renderer = make_renderer()

    global TRAIN_MASKS
    TRAIN_MASKS = load_train_masks()

    cond_75  = raw_tokens[75].unsqueeze(0).to(DEVICE)
    gm_tight = tight_gt_mask(75)
    print(f'\n[MASK] tight GT mask: {int(gm_tight.sum()):,} px = '
          f'{float(gm_tight.float().mean())*100:.1f}% of frame')

    # ── frozen reference + aligned static render mask ─────────────────────────
    frozen_ref = _frozen_reference(flow_model, dec_model, renderer,
                                   fixed_noise_feats, coords, cond_75, gm_tight)
    render_mask = frozen_ref['sil']
    render_mask_stale = load_render_mask()
    _iou_ms = float((render_mask & render_mask_stale).sum()
                    / max(int((render_mask | render_mask_stale).sum()), 1))
    print(f'  stale mask.png : {int(render_mask_stale.sum()):,} px')
    print(f'  this run mask  : {int(render_mask.sum()):,} px   '
          f'IoU(stale,this)={_iou_ms:.4f}')
    print(f'  frozen verts   : {frozen_ref["verts"]:,}   '
          f'silhouette IoU vs GT = {frozen_ref["iou_gt"]:.4f}', flush=True)
    print(f'  NOISE FLOOR    : two identical frozen runs differ by '
          f'{frozen_ref["verts_noise"]} verts '
          f'({frozen_ref["verts_noise_pct"]:.4f}%), silhouette IoU '
          f'{frozen_ref["iou_noise"]:.5f}')
    print(f'                   geometry drift below this is the pipeline, '
          f'not the adapter.', flush=True)

    # ── GATE-align ────────────────────────────────────────────────────────────
    if USE_ALIGN:
        slat_a = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_75, None)
        with torch.no_grad():
            mesh_a = dec_model(slat_a)[0]
        _, m_un = render_mesh(mesh_a, renderer, aligned=False)
        un = (m_un.squeeze(0) > 0.5)
        iou_un = int((un & gm_tight).sum()) / max(int((un | gm_tight).sum()), 1)
        iou_al = frozen_ref['iou_gt']
        print(f'\n[GATE-align] silhouette IoU vs GT at frame 75')
        print(f'  unaligned = {iou_un:.4f}     aligned = {iou_al:.4f}   '
              f'(+{iou_al - iou_un:.4f})', flush=True)
        assert iou_al > iou_un + 0.05, (
            f'GATE-align FAILED: {iou_un:.4f} -> {iou_al:.4f}. The transform is '
            f'not reaching the renderer, or alignment.json is wrong.')
        print('[GATE-align] PASSED', flush=True)
        json.dump({'iou_unaligned': iou_un, 'iou_aligned': iou_al,
                   'scale': ALIGN_S, 'rotvec': ALIGN_ROTVEC, 't': ALIGN_T},
                  open(_LOGD / 'alignment_applied.json', 'w'), indent=2)
        del slat_a, mesh_a, m_un, un
        gc.collect(); torch.cuda.empty_cache()

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None

    # ── GATE-identity: at B=0 the patched path must equal the frozen path ─────
    if not resumed:
        print('\n[GATE-identity] B=0 => patched flow == frozen flow...', flush=True)
        # TRELLIS's own forward is NOT bit-reproducible: xformers attention and
        # spconv's `native` algo both accumulate with atomics, so two identical
        # frozen runs disagree. Observed across two gate jobs on the same A40:
        # frozen vertex count 215,482 vs 215,492 with every seed fixed.
        # So "patched == frozen" cannot be tested against zero. Measure the
        # model's OWN run-to-run spread first, then require the patch to sit
        # inside it. Testing against a hardcoded tolerance would either pass a
        # broken patch or fail a correct one, depending on the constant picked.
        slat_f1 = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_75, None)
        f1 = slat_f1.feats.clone(); del slat_f1
        gc.collect(); torch.cuda.empty_cache()

        slat_f2 = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_75, None)
        f2 = slat_f2.feats.clone(); del slat_f2
        gc.collect(); torch.cuda.empty_cache()

        slat_l = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_75, registry)
        fl = slat_l.feats.clone(); del slat_l
        gc.collect(); torch.cuda.empty_cache()

        d_noise = (f1 - f2).abs().max().item()      # frozen vs frozen
        d_patch = (f1 - fl).abs().max().item()      # frozen vs patched at B=0
        scale   = f1.abs().max().item()
        tol     = max(3.0 * d_noise, 1e-6)
        print(f'  SLaT dynamic range        : {scale:.3f}')
        print(f'  frozen vs frozen  (noise) : {d_noise:.3e}  '
              f'<- the model is not deterministic')
        print(f'  frozen vs patched (B=0)   : {d_patch:.3e}  '
              f'(tol = 3x noise = {tol:.3e})')
        assert d_patch <= tol, (
            f'GATE-identity FAILED: the patch adds {d_patch:.3e}, which is '
            f'{d_patch / max(d_noise, 1e-30):.1f}x the model\'s own run-to-run '
            f'noise of {d_noise:.3e}. That is a logic difference, not float '
            f'jitter — the patched forward is not reproducing frozen '
            f'cross-attention.')
        print('[GATE-identity] PASSED', flush=True)
        json.dump({'d_noise': d_noise, 'd_patch': d_patch, 'tol': tol,
                   'slat_absmax': scale},
                  open(_LOGD / 'gate_identity.json', 'w'), indent=2)
        del f1, f2, fl
        gc.collect(); torch.cuda.empty_cache()

    # ── optimizer + resume ────────────────────────────────────────────────────
    trainable = list(registry.parameters())
    optimizer = torch.optim.Adam(trainable, lr=LR, eps=1e-16, weight_decay=0.0)
    start_epoch, history, best_psnr = 1, [], -float('inf')

    if resumed:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ck = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        assert ck.get('run_id') == RUN_ID, \
            f'run_id mismatch: {ck.get("run_id")} != {RUN_ID}'
        registry.load_state_dict(ck['registry_state'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        best_psnr   = ck.get('best_psnr', -float('inf'))
        loss_scale  = ck.get('loss_scale', LOSS_SCALE0)
        if ck.get('rng_state') is not None:
            _RNG.bit_generator.state = ck['rng_state']
            print('  knot RNG restored — the sequence continues, not restarts')
        else:
            print('  WARNING: checkpoint has no rng_state; knot sequence restarts')
        hp = _OUT / 'loss_history.json'
        if hp.exists():
            history = json.load(open(hp))
        print(f'  resumed from epoch {ck["epoch"]}  loss_scale={loss_scale:.0f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    # ── GATE-grad: refuse to train if any B matrix is starved ─────────────────
    if not resumed:
        # rung15 trains at EVERY knot, so a single-k probe is not a gate — it is
        # a spot check. Gradient starvation in a flow model is timestep
        # dependent: at high t the latent is nearly pure noise and the render is
        # almost content-free, which is exactly where the signal can collapse.
        # Probe the ends and the middle of the pool, and require all of them.
        pool = _TS_POOL
        probe_ks = sorted({pool[0], pool[len(pool)//2], pool[-1]})
        print(f'\n[GATE-grad] gradient reachability at k = {probe_ks} '
              f'(pool = {pool[0]}..{pool[-1]})', flush=True)
        gate_rec, peak = {}, 0.0
        for pk in probe_ks:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            probe = _train_step(flow_model, dec_model, registry, renderer,
                                lpips_fn, fixed_noise_feats, coords, raw_tokens,
                                75, render_mask, loss_scale, trainable,
                                optimizer, do_step=False, k=pk)
            assert probe is not None, (
                f'GATE-grad FAILED at k={pk}: the probe step returned nothing — '
                f'either no gradient reached the SLaT leaf, or the grads were '
                f'non-finite. See the f075 line above for which.')
            st = probe['gstats']
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            print(f'  k={pk:02d} t={probe["t_k"]:.4f}  loss={probe["total"]:.5f}  '
                  + '  '.join(f'{a}={b:.3e}' for a, b in st['per_target'].items())
                  + f'  n_zero={st["n_zero"]}/{st["n_total"]}', flush=True)
            dead = [b for b, v in sorted(st['per_block'].items()) if v == 0.0]
            assert st['n_zero'] == 0, (
                f'GATE-grad FAILED at k={pk} (t={probe["t_k"]:.4f}): '
                f'{st["n_zero"]}/{st["n_total"]} B matrices get zero gradient. '
                f'Dead blocks: {dead}. rung15 trains at this knot, so this would '
                f'be a silent no-op for part of the schedule. Restrict the pool '
                f'with --tsample late, or raise LOSS_SCALE.')
            gate_rec[str(pk)] = {'t': probe['t_k'], 'loss': probe['total'], **st}
        print(f'  peak GPU over the probes: {peak:.2f} GB')
        print('[GATE-grad] PASSED\n', flush=True)
        optimizer.zero_grad(set_to_none=True)
        gate_rec['peak_gpu_gb'] = peak
        gate_rec['pool'] = pool
        json.dump(gate_rec, open(_LOGD / 'gate_grad.json', 'w'), indent=2)

    # ── GATE-sat: silent on frozen, active past tau ──────────────────────────
    if not resumed:
        _z = torch.zeros(64, 101, device=DEVICE)
        _z[:, 53:101] = 2.94                      # frozen's measured MAXIMUM logit
        _p_frozen = float(sat_penalty(_z, args.tau))
        _z[:, 53:101] = 5.71                      # the adapter's measured maximum
        _p_adapt = float(sat_penalty(_z, args.tau))
        print(f'\n[GATE-sat] tau={args.tau}  lam={args.lam}')
        print(f'  penalty at frozen max logit 2.94 : {_p_frozen:.6f}')
        print(f'  penalty at adapted max      5.71 : {_p_adapt:.6f}')
        assert _p_frozen == 0.0, (
            f'GATE-sat FAILED: the penalty charges frozen ({_p_frozen:.6f}). tau '
            f'is below what frozen TRELLIS itself produces, so this would fight '
            f'the base model rather than only saturation.')
        if args.lam > 0:
            assert _p_adapt > 0.0, 'GATE-sat FAILED: penalty inert past tau'
        print('[GATE-sat] PASSED'
              + ('  (lam=0 -> control, reproduces rung16)' if args.lam == 0 else ''),
              flush=True)
        json.dump({'tau': args.tau, 'lam': args.lam,
                   'penalty_at_frozen_max': _p_frozen,
                   'penalty_at_adapted_max': _p_adapt},
                  open(_LOGD / 'gate_sat.json', 'w'), indent=2)

    # ── GATE-mask: the denominator must actually have changed ────────────────
    if not resumed:
        _lk = torch.stack([load_gt(i)[1] for i in (1, 75, 150)]).float().mean()
        _tm = TRAIN_MASKS[[0, 74, 149]].float().mean()
        _ratio = float(_lk / _tm.clamp(min=1e-8))
        print(f'\n[GATE-mask] loss denominator')
        print(f'  rung15 leaky mask : {100*float(_lk):.2f}% of the frame')
        print(f'  this run          : {100*float(_tm):.2f}%')
        print(f'  MSE multiplied by : {_ratio:.2f}x  '
              f'-> MSE:LPIPS moves from ~1.8:1 to ~{1.8*_ratio:.1f}:1', flush=True)
        if args.train_mask == 'leaky':
            assert abs(_ratio - 1.0) < 0.01, 'leaky mode must reproduce rung15'
            print('[GATE-mask] PASSED (control: identical to rung15)', flush=True)
        else:
            assert _ratio > 4.0, (
                f'GATE-mask FAILED: denominator only changed {_ratio:.2f}x. The '
                f'mask is not tight, so this run is not the experiment it claims.')
            print('[GATE-mask] PASSED', flush=True)
        json.dump({'leaky_frac': float(_lk), 'train_frac': float(_tm),
                   'mse_multiplier': _ratio, 'train_mask': args.train_mask},
                  open(_LOGD / 'gate_mask.json', 'w'), indent=2)

    if args.gates_only:
        print('[GATES-ONLY] every gate passed. Nothing trained; no checkpoint '
              'written, so the real run starts clean.', flush=True)
        # A stray e000 checkpoint would make the real run think it is resuming
        # and silently skip its own gates.
        for _p in _CKPT.glob('lora_e*.pt'):
            _p.unlink()
        return

    # ── training ──────────────────────────────────────────────────────────────
    csv_path = _LOGD / 'epoch_metrics.csv'
    if not csv_path.exists():
        csv_path.write_text(
            'epoch,loss_total,loss_mse,loss_lpips,held_psnr,held_std,held_ssim,'
            'held_lpips,B_norm_mean,B_norm_max,g_to_q,g_to_kv,g_to_out,n_zero,'
            'geom_verts,geom_verts_delta_pct,geom_iou_vs_frozen,geom_iou_vs_gt,'
            'time_s\n')
    geo_csv = _LOGD / 'geometry_drift.csv'
    if not geo_csv.exists():
        geo_csv.write_text('epoch,verts,verts_frozen,verts_delta_pct,area_px,'
                           'area_frozen_px,iou_vs_frozen,iou_vs_gt,'
                           'iou_frozen_vs_gt\n')

    baseline_cache = {}
    print(f'[TRAIN] epochs {start_epoch}->{EPOCHS}  frames/epoch={len(TRAIN)}  '
          f'LOSS_SCALE={loss_scale:.0f}', flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        ep_t0 = time.time()
        order = TRAIN[:]
        random.shuffle(order)
        tot_loss = tot_mse = tot_lp = tot_sat = 0.0
        n_steps  = 0
        last_st  = None

        print(f'\n[EPOCH {epoch}/{EPOCHS}] starting...', flush=True)

        # Loss is NOT comparable across knots — at high t the render is nearly
        # content-free and the loss is large for reasons that have nothing to do
        # with the adapter. So the epoch mean is a mix of apples and oranges and
        # must not be read as "training progress". Held-out PSNR (full 25-step
        # ODE) is the only comparable number. Per-knot losses are bucketed so
        # the mixture is visible instead of hidden inside the average.
        k_loss, k_count = {}, {}
        n_skip, skip_k = 0, {}

        for fi in order:
            r = _train_step(flow_model, dec_model, registry, renderer, lpips_fn,
                            fixed_noise_feats, coords, raw_tokens, fi,
                            render_mask, loss_scale, trainable, optimizer,
                            do_step=True)
            if r is None:
                n_skip += 1
                skip_k[_LAST_K[0]] = skip_k.get(_LAST_K[0], 0) + 1
                continue
            tot_loss += r['total']; tot_mse += r['mse']; tot_lp += r['lp']
            tot_sat  += r.get('sat', 0.0)
            n_steps  += 1
            last_st   = r['gstats']
            kk = r['k']
            k_loss[kk]  = k_loss.get(kk, 0.0) + r['total']
            k_count[kk] = k_count.get(kk, 0) + 1

            if n_steps % 15 == 0:
                print(f'  e{epoch:02d} [{n_steps:03d}/{len(TRAIN)}] f{fi:03d} '
                      f'k{kk:02d} t={r["t_k"]:.3f}  mse={r["mse"]:.5f}  '
                      f'lpips={r["lp"]:.5f}  total={r["total"]:.5f}  '
                      f'nz={r["gstats"]["n_zero"]}', flush=True)

        # ── epoch eval ────────────────────────────────────────────────────────
        held = evaluate_frames(flow_model, dec_model, registry, renderer,
                               render_mask, lpips_fn, fixed_noise_feats, coords,
                               raw_tokens, HELD_OUT)
        geo  = measure_geometry_drift(flow_model, dec_model, registry, renderer,
                                      fixed_noise_feats, coords, cond_75,
                                      frozen_ref, gm_tight)
        bn   = b_norms(registry)
        ep_loss = tot_loss / max(n_steps, 1)
        ep_mse  = tot_mse  / max(n_steps, 1)
        ep_lp   = tot_lp   / max(n_steps, 1)
        ep_sat  = tot_sat  / max(n_steps, 1)
        ep_time = time.time() - ep_t0
        new_best = held['psnr_mean'] > best_psnr
        if new_best:
            best_psnr = held['psnr_mean']

        pt = (last_st or {}).get('per_target', {})
        print(f'\n[EPOCH {epoch}/{EPOCHS}]  loss={ep_loss:.5f}  '
              f'(mse={ep_mse:.5f}  lpips x0.1={ep_lp*W_LPIPS:.5f})  '
              f'<- mixed across knots, NOT a progress metric', flush=True)
        print(f'  sat penalty  raw={ep_sat:.6f}  x lam({args.lam})={args.lam*ep_sat:.6f}'
              f'   ({100*args.lam*ep_sat/max(ep_loss,1e-12):.1f}% of the loss)'
              f'{"   <- ZERO: no logit exceeded tau" if ep_sat==0 else ""}', flush=True)
        if n_skip:
            print(f'  WARNING: {n_skip}/{len(TRAIN)} steps SKIPPED this epoch '
                  f'(non-finite grad or no grad at the leaf). The epoch mean is '
                  f'computed over the survivors only.', flush=True)
            print(f'           skipped knots: '
                  + '  '.join(f'k{k}:{n}' for k, n in sorted(skip_k.items()))
                  + '   <- if these cluster at one end of the schedule, the pool '
                    'is wrong, not the adapter', flush=True)
        if k_count:
            ks = sorted(k_count)
            print('  per-knot mean loss  ' + '  '.join(
                f'k{k}({k_count[k]}):{k_loss[k]/k_count[k]:.4f}'
                for k in ks[:6]) + (' ...' if len(ks) > 6 else ''), flush=True)
            json.dump({str(k): {'t': float(T_PAIRS[k][0]), 'n': k_count[k],
                                'mean_loss': k_loss[k] / k_count[k]}
                       for k in ks},
                      open(_LOGD / f'knot_loss_e{epoch:03d}.json', 'w'), indent=2)
        print(f'  ||B||  mean={np.mean(bn):.4f}  max={np.max(bn):.4f}', flush=True)
        print(f'  ||dB|| ' + '  '.join(f'{k}={v:.3e}' for k, v in pt.items()) +
              f'   n_zero={(last_st or {}).get("n_zero", -1)}', flush=True)
        print(f'  held   PSNR={held["psnr_mean"]:.3f}+-{held["psnr_std"]:.3f}  '
              f'SSIM={held["ssim_mean"]:.4f}  LPIPS={held["lpips_mean"]:.4f}  '
              + ('BEST' if new_best else '') + f'  time={ep_time:.1f}s', flush=True)
        print(f'  GEOM   verts={geo["verts"]:,} '
              f'({geo["verts_delta_pct"]:+.2f}% vs frozen)  '
              f'IoU(vs frozen)={geo["iou_vs_frozen"]:.4f}  '
              f'IoU(vs GT)={geo["iou_vs_gt"]:.4f} '
              f'[frozen was {geo["iou_frozen_vs_gt"]:.4f}]', flush=True)

        ckpt = {'epoch': epoch, 'run_id': RUN_ID, 'best_psnr': best_psnr,
                'loss_scale': loss_scale, 'optimizer': optimizer.state_dict(),
                'registry_state': registry.state_dict(),
                # Without this, a resume restarts _RNG from args.seed and
                # replays the SAME knot sequence it already trained on. Across
                # a 3-job chain that would concentrate training on whichever
                # knots the first epochs happened to draw.
                'rng_state': _RNG.bit_generator.state}
        torch.save(ckpt, _CKPT / f'lora_e{epoch:03d}.pt')
        if new_best:
            torch.save(ckpt, _CKPT / 'lora_best.pt')
            print('  [CKPT] new best -> lora_best.pt')

        rec = {'epoch': epoch, 'loss_total': ep_loss, 'loss_mse': ep_mse,
               'loss_lpips': ep_lp, 'held_psnr': held['psnr_mean'],
               'held_std': held['psnr_std'], 'held_ssim': held['ssim_mean'],
               'held_lpips': held['lpips_mean'],
               'B_norm_mean': float(np.mean(bn)), 'B_norm_max': float(np.max(bn)),
               'geom_verts': geo['verts'],
               'geom_verts_delta_pct': geo['verts_delta_pct'],
               'geom_iou_vs_frozen': geo['iou_vs_frozen'],
               'geom_iou_vs_gt': geo['iou_vs_gt'],
               'grad_per_target': pt, 'n_zero': (last_st or {}).get('n_zero', -1),
               'time_s': ep_time}
        history.append(rec)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)

        with open(csv_path, 'a') as fh:
            fh.write(f'{epoch},{ep_loss:.6f},{ep_mse:.6f},{ep_lp:.6f},'
                     f'{held["psnr_mean"]:.4f},{held["psnr_std"]:.4f},'
                     f'{held["ssim_mean"]:.5f},{held["lpips_mean"]:.5f},'
                     f'{np.mean(bn):.5f},{np.max(bn):.5f},'
                     f'{pt.get("to_q", 0):.4e},{pt.get("to_kv", 0):.4e},'
                     f'{pt.get("to_out", 0):.4e},'
                     f'{(last_st or {}).get("n_zero", -1)},'
                     f'{geo["verts"]},{geo["verts_delta_pct"]:.3f},'
                     f'{geo["iou_vs_frozen"]:.5f},{geo["iou_vs_gt"]:.5f},'
                     f'{ep_time:.1f}\n')
        with open(geo_csv, 'a') as fh:
            fh.write(f'{epoch},{geo["verts"]},{geo["verts_frozen"]},'
                     f'{geo["verts_delta_pct"]:.3f},{geo["area_px"]},'
                     f'{geo["area_frozen_px"]},{geo["iou_vs_frozen"]:.5f},'
                     f'{geo["iou_vs_gt"]:.5f},{geo["iou_frozen_vs_gt"]:.5f}\n')
        save_curves(history)

        if epoch % DIAG_EVERY == 0:
            save_diagnostics(flow_model, dec_model, registry, renderer, epoch,
                             fixed_noise_feats, coords, raw_tokens, baseline_cache)

    # ── final eval ────────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('[FINAL EVAL]')
    print('=' * 72)
    best_path = _CKPT / 'lora_best.pt'
    if best_path.exists():
        bc = torch.load(best_path, map_location=DEVICE, weights_only=True)
        registry.load_state_dict(bc['registry_state'])
        print(f'  Loaded best (epoch={bc["epoch"]}  psnr={bc["best_psnr"]:.3f})')

    res_held  = evaluate_frames(flow_model, dec_model, registry, renderer,
                                render_mask, lpips_fn, fixed_noise_feats, coords,
                                raw_tokens, HELD_OUT)
    res_train = evaluate_frames(flow_model, dec_model, registry, renderer,
                                render_mask, lpips_fn, fixed_noise_feats, coords,
                                raw_tokens, TRAIN)
    geo_final = measure_geometry_drift(flow_model, dec_model, registry, renderer,
                                       fixed_noise_feats, coords, cond_75,
                                       frozen_ref, gm_tight)

    print(f'  Held:  PSNR={res_held["psnr_mean"]:.3f}+-{res_held["psnr_std"]:.3f}  '
          f'SSIM={res_held["ssim_mean"]:.4f}  LPIPS={res_held["lpips_mean"]:.4f}')
    print(f'  Train: PSNR={res_train["psnr_mean"]:.3f}')
    print(f'  GEOM:  {geo_final["verts"]:,} verts '
          f'({geo_final["verts_delta_pct"]:+.2f}%)  '
          f'IoU vs frozen={geo_final["iou_vs_frozen"]:.4f}')
    print(f'\n  reference: rung13 (decoder-side, aligned) held PSNR = 22.646')
    print(f'             rung2  (cross-attn, UNaligned) held PSNR = 17.362')

    json.dump({'run_id': RUN_ID, 'label': _LABEL,
               'held': res_held, 'train': res_train, 'geometry': geo_final,
               'reference': {'rung13_held_psnr': 22.646, 'rung2_held_psnr': 17.362}},
              open(_OUT / 'final_eval.json', 'w'), indent=2)
    print(f'\n[DONE] best held PSNR={best_psnr:.3f} dB   output={_OUT}')


def _train_step(flow_model, dec_model, registry, renderer, lpips_fn,
                noise_feats, coords, raw_tokens, fi, render_mask,
                loss_scale, trainable, optimizer, do_step: bool, k=None):
    """
    One frame at ONE randomly chosen schedule knot k.

    THE RUNG-15 CHANGE is the single line `k = sample_k()`. Everything else is
    rung14-v1's mechanics: the two-stage backward keeps the flow graph and the
    decoder graph from co-existing (flow_model is ~4 GB), and the LoRA stays
    active through the ungraded prefix so x_k is the state the adapted model
    actually reaches.

    k is an argument so the gates can pin it and so --tsample last reproduces
    rung14-v1 exactly.
    """
    if k is None:
        k = sample_k()
    _LAST_K[0] = k          # so a None return can still be attributed to a knot
    t_k, t_next = T_PAIRS[k]

    cond = raw_tokens[fi].unsqueeze(0).to(DEVICE)
    optimizer.zero_grad(set_to_none=True)

    noise_sp = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
    x_k = denoise_prefix_to_k(flow_model, noise_sp, cond, registry, k)

    # --- pass 1: no_grad, to get a detached SLaT leaf for the decoder half ---
    with torch.no_grad():
        v_ng    = flow_eval_at(flow_model, x_k, t_k, cond, registry, False)
        x0_ng   = pred_to_xstart(x_k, t_k, v_ng)
        slat_ng = normalize_slat(x0_ng)

    leaf      = slat_ng.feats.detach().clone().requires_grad_(True)
    slat_leaf = slat_ng.replace(leaf)

    # stage 1 — decoder + render + backward, flow parked if the GPU is small
    vals = None
    with parked(flow_model):
        # Capture out_layer's raw output from THIS graded decode so the penalty's
        # gradient reaches the LoRA through the same path the render loss does.
        _RAW.clear()
        def _cap(m, i, o): _RAW['f'] = o.feats
        _h = dec_model.out_layer.register_forward_hook(_cap)
        try:
            mesh = dec_model(slat_leaf)[0]
        finally:
            _h.remove()
        assert 'f' in _RAW, 'out_layer hook did not fire'
        color, _ = render_mesh(mesh, renderer)
        gt, _leaky = load_gt(fi)
        # THE RUNG-16 CHANGE. _leaky is what rung15 used here and is discarded;
        # evaluate_frames still uses it, so the reported metrics stay comparable.
        gt_mask = TRAIN_MASKS[fi - 1]
        mse, lp, total = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)
        sat = sat_penalty(_RAW['f'], args.tau)
        total = total + args.lam * sat
        (total * loss_scale).backward()
        vals = {'mse': mse.item(), 'lp': lp.item(), 'total': total.item(),
                'sat': sat.item()}
        _RAW.clear()
    if vals is None:
        raise RuntimeError(f'frame {fi}: decode/render/backward failed')

    if leaf.grad is None:
        print(f'  f{fi:03d} k{k:02d} SKIP: no grad reached the SLaT leaf', flush=True)
        optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        return None

    grad_slat = leaf.grad.detach()
    del mesh, color, gt, gt_mask, mse, lp, total, slat_leaf, leaf
    gc.collect(); torch.cuda.empty_cache()

    # stage 2 — replay the SAME knot k with grad, inject the SLaT gradient.
    # Must be the same k and the same x_k, or the gradient is being applied to a
    # different function than the one that produced it.
    v_g    = flow_eval_at(flow_model, x_k, t_k, cond, registry, True)
    x0_g   = pred_to_xstart(x_k, t_k, v_g)
    slat_r = normalize_slat(x0_g)
    torch.autograd.backward(slat_r.feats, grad_slat)

    for p in trainable:
        if p.grad is not None:
            p.grad.div_(loss_scale)

    gstats = b_grad_stats(registry)

    bad = any(p.grad is not None and not torch.isfinite(p.grad).all()
              for p in trainable)
    if bad:
        print(f'  f{fi:03d} k{k:02d} non-finite grad — step skipped', flush=True)
        optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        return None

    if do_step:
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()

    del v_g, x0_g, slat_r, grad_slat, x_k, v_ng, x0_ng, slat_ng, noise_sp, cond
    gc.collect(); torch.cuda.empty_cache()
    return {**vals, 'gstats': gstats, 'k': k, 't_k': float(t_k)}


if __name__ == '__main__':
    main()
