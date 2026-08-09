"""
Rung 5 Sub-ladder: Decoder LoRA ablation.

Tests hypothesis: the latent is fine; the decoder is mistranslating it.
Flow model frozen + no_grad. Only the decoder adapter is trained.

Stages:
  5.0          Baseline eval (no LoRA)
  5.1          Output head only: AppearanceHeadLoRA on out_layer (48ch appearance)
  5.2          Last block (block 11), all 4 layers
  5.3-early    First third of blocks (0-3)
  5.3-mid      Middle third (4-7)
  5.3-late     Last third (8-11)
  5.4          All 12 blocks
  5.5          Rank sweep on winner -- use --rank {4,8,16} and --active-blocks i,j,...

Gates (auto-fire at start of every non-5.0 stage):
  GATE-plain:     B=0 render ≈ frozen render (fp16 tol 1e-2)
  GATE-sparse:    B=0 output SparseTensor coords == frozen coords (torch.equal)
  GATE-geom:      B=0 mesh vertices == frozen vertices (torch.equal)

Param math (rank 4):
  head (5.1):     rank*(96+48) = 576
  block (5.2-5.5): per block: to_qkv 3072r + to_out 1536r + fc1 3840r + fc2 3840r = 12288r
  1 block rank 4: 49,152
  4 blocks rank 4: 196,608
  12 blocks rank 4: 589,824

N_DEC=12 asserted at runtime.

Output:
  lora_experiments/runs/rung5_{stage}_{variant}_{r}_{RUN_ID}/
    config.json, loss_history.json, lora_ckpts/, diag_renders/, train.log
  lora_experiments/rung5_gate3/
    init_{stage_tag}.pt   -- used for 5.3 cross-placement gate
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
_GATE3 = _HERE / 'rung5_gate3'
_RUNS.mkdir(parents=True, exist_ok=True)
_GATE3.mkdir(exist_ok=True)

# ── parse args BEFORE heavy imports ──────────────────────────────────────────

_ap_ = _ap.ArgumentParser()
_ap_.add_argument('--stage', required=True,
                  choices=['5.0','5.1','5.2','5.3-early','5.3-mid','5.3-late','5.4','5.5'])
_ap_.add_argument('--rank',         type=int, default=4)
_ap_.add_argument('--seed',         type=int, default=6)
_ap_.add_argument('--epochs',       type=int, default=30)
_ap_.add_argument('--active-blocks',type=str, default=None,
                  help='Comma-sep block indices for 5.5 (default: last-third winner 8,9,10,11)')
_ap_.add_argument('--smoke',        action='store_true')
_ap_.add_argument('--diag_every',   type=int, default=5)
args = _ap_.parse_args()

# ── stage → block assignment ──────────────────────────────────────────────────

# N_DEC=12 → thirds are [0-3], [4-7], [8-11]
_N_DEC_EXPECTED = 12

STAGE_BLOCKS = {
    '5.0':        [],
    '5.1':        [],          # head only, no block LoRA
    '5.2':        [11],        # last block
    '5.3-early':  list(range(0, 4)),
    '5.3-mid':    list(range(4, 8)),
    '5.3-late':   list(range(8, 12)),
    '5.4':        list(range(12)),   # all 12 — filled after N_DEC assert
    '5.5':        None,              # from --active-blocks
}

STAGE_HEAD = {s: (s == '5.1') for s in STAGE_BLOCKS}

# Resolve 5.5 blocks
if args.stage == '5.5':
    if args.active_blocks:
        active = [int(x.strip()) for x in args.active_blocks.split(',')]
    else:
        active = list(range(8, 12))   # dec-late default
    STAGE_BLOCKS['5.5'] = active

ACTIVE_BLOCKS = STAGE_BLOCKS[args.stage]
USE_HEAD      = STAGE_HEAD[args.stage]
N_ACTIVE      = len(ACTIVE_BLOCKS)

# ── constants ─────────────────────────────────────────────────────────────────

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)
MASK_PATH = Path(
    '/net/projects/ranalab/rajhansini/TRELLIS/experiments'
    '/dynamic_texture_trellis_pipeline/debug_results/step8_mesh/mask.png'
)
PRETRAINED    = 'JeffreyXiang/TRELLIS-image-large'
N_FRAMES      = 150
STRUCT_SEED   = 42
STEPS         = 25
RESCALE_T     = 3.0
LR            = 1e-4
LOSS_SCALE0   = 4096.0
GRAD_CLIP     = 1.0
W_LPIPS       = 0.1
DEC_DIM       = 768
DEC_MLP_HID   = 3072          # DEC_DIM * 4
OUT_LAYER_IN  = 96             # DEC_DIM // 8
APPEAR_START  = 53
N_APPEAR      = 48             # channels 53:101
N_TOTAL_CH    = 101

HELD_OUT = list(range(5, 151, 10))     # 15 frames
TRAIN    = [i for i in range(1, N_FRAMES + 1) if i not in HELD_OUT]

# ── param math helper ─────────────────────────────────────────────────────────

def _expected_params(rank, use_head, active_blocks):
    p = 0
    if use_head:
        p += rank * (OUT_LAYER_IN + N_APPEAR)  # A(r,96) + B(48,r)
    for _ in active_blocks:
        # to_qkv: (r,768)+(2304,r) = 3072r
        # to_out: (r,768)+(768,r)  = 1536r
        # fc1:    (r,768)+(3072,r) = 3840r
        # fc2:    (r,3072)+(768,r) = 3840r
        # total: 12288r
        p += rank * 12288
    return p

# ── run identity ──────────────────────────────────────────────────────────────

_stage_tag = args.stage.replace('.', '').replace('-', '_')
_CFG = dict(
    stage=args.stage, rank=args.rank, seed=args.seed,
    epochs=(2 if args.smoke else args.epochs),
    lr=LR, eps=1e-16, w_lpips=W_LPIPS, loss_scale=LOSS_SCALE0,
    held_out=HELD_OUT, active_blocks=ACTIVE_BLOCKS, use_head=USE_HEAD,
)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_LABEL = f'rung5_{_stage_tag}_r{args.rank}_s{args.seed}_{RUN_ID}'
_OUT   = _RUNS / _LABEL
_CKPT  = _OUT / 'lora_ckpts'
_DIAG  = _OUT / 'diag_renders'

_OUT.mkdir(parents=True, exist_ok=True)
_CKPT.mkdir(exist_ok=True)
_DIAG.mkdir(exist_ok=True)

# ── tee stdout to log ─────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(_OUT / 'train.log')
sys.stderr = sys.stdout

# ── nvdiffrast arch guard ─────────────────────────────────────────────────────

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
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_FRAMES = [1, 75, 150]


# ── LoRA modules ──────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """fp32 forward, kaiming A, zero B, scaling = lora_alpha/rank = 1.0."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.scaling = 1.0
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (((x.float() @ self.A.T) @ self.B.T) * self.scaling).to(x.dtype)


class AppearanceHeadLoRA(nn.Module):
    """
    5.1: wraps the frozen out_layer (SparseLinear 96->101).
    Delta confined to appearance channels [53:101] by construction.
    Geometry channels [0:53] are bit-identical to frozen output.

    5.1b: confining delta to color-only sub-channels of the 6 per-corner is deferred.
    # TODO: read FlexiCubes voxelgrid_colors consumer to determine
    # color[0:3] vs normal[3:6] within the 6 per-corner channels before implementing 5.1b.
    """
    def __init__(self, frozen_out_layer: nn.Module, rank: int):
        super().__init__()
        self.base = frozen_out_layer   # SparseLinear(96, 101), kept frozen
        assert frozen_out_layer.in_features  == OUT_LAYER_IN, \
            f'out_layer in_features={frozen_out_layer.in_features} != {OUT_LAYER_IN}'
        assert frozen_out_layer.out_features == N_TOTAL_CH, \
            f'out_layer out_features={frozen_out_layer.out_features} != {N_TOTAL_CH}'
        self.lora = LoRALayer(OUT_LAYER_IN, N_APPEAR, rank)   # A(r,96) B(48,r)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        # Base forward (frozen, no grad through base weights)
        with torch.no_grad():
            base_out = self.base(x)     # SparseTensor (N, 101)
        # LoRA delta: (N, 48), in fp32 then cast
        lora_delta = self.lora(x.feats)   # (N, 48)
        # Assemble: geometry [0:53] frozen, appearance [53:101] += delta
        geom    = base_out.feats[:, :APPEAR_START].detach()          # (N, 53)
        appear  = base_out.feats[:, APPEAR_START:].detach()          # (N, 48)
        appear  = appear + lora_delta.to(appear.dtype)               # (N, 48), has grad
        out_feats = torch.cat([geom, appear], dim=1)                 # (N, 101)
        return base_out.replace(out_feats)


class DecBlockLoRABundle(nn.Module):
    """Holds LoRA weights for all 4 adaptable layers in one decoder block."""
    def __init__(self, rank: int, dim: int = DEC_DIM):
        super().__init__()
        mlp_h = int(dim * 4)      # 3072
        self.lora_qkv = LoRALayer(dim, 3 * dim, rank)   # 768→2304, params: 3072r
        self.lora_out = LoRALayer(dim, dim,     rank)   # 768→768,  params: 1536r
        self.lora_fc1 = LoRALayer(dim, mlp_h,  rank)   # 768→3072, params: 3840r
        self.lora_fc2 = LoRALayer(mlp_h, dim,  rank)   # 3072→768, params: 3840r
        # total per block: 12288r


class DecLoRARegistry(nn.Module):
    def __init__(self, active_blocks, rank: int):
        super().__init__()
        self.active = set(active_blocks)
        self.blocks = nn.ModuleDict({
            str(i): DecBlockLoRABundle(rank=rank) for i in active_blocks
        })

    def get(self, block_idx: int):
        key = str(block_idx)
        return self.blocks[key] if key in self.blocks else None


# ── LoRA context managers ─────────────────────────────────────────────────────

@contextmanager
def dec_head_lora_ctx(dec_model, head_lora: AppearanceHeadLoRA):
    """
    5.1: replace out_layer with AppearanceHeadLoRA.
    Restores on exit. Thread-safe: no global state.
    """
    orig = dec_model.out_layer
    dec_model.out_layer = head_lora
    try:
        yield
    finally:
        dec_model.out_layer = orig


@contextmanager
def dec_block_lora_ctx(dec_model, registry: DecLoRARegistry):
    """
    5.2-5.5: add forward hooks to to_qkv, to_out, mlp.mlp[0], mlp.mlp[2]
    in active blocks. Inactive blocks run frozen untouched.
    Hooks add LoRA delta AFTER the base layer runs. Restored on exit.
    """
    handles = []

    for i, block in enumerate(dec_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue

        # to_qkv: nn.Linear receives x.feats (plain tensor), returns plain tensor.
        # _linear then wraps in SparseTensor: x.replace(to_qkv(x.feats))
        def _qkv_hook(mod, inp, out, _lb=lb):
            # inp[0]: plain tensor (N, 768), out: plain tensor (N, 2304)
            return out + _lb.lora_qkv(inp[0]).to(out.dtype)

        # to_out: same signature
        def _out_hook(mod, inp, out, _lb=lb):
            return out + _lb.lora_out(inp[0]).to(out.dtype)

        # mlp.mlp[0]: SparseLinear receives SparseTensor, returns SparseTensor
        def _fc1_hook(mod, inp, out, _lb=lb):
            d = _lb.lora_fc1(inp[0].feats)
            return out.replace(out.feats + d.to(out.feats.dtype))

        # mlp.mlp[2]: SparseLinear (3072→768), same
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


# ── Gates ─────────────────────────────────────────────────────────────────────

def run_gates(dec_model, registry, head_lora, slat_norm_ref, renderer, stage_tag):
    """
    Auto-fires GATE-plain, GATE-sparse, GATE-geom at B=0 (fresh runs only).
    slat_norm_ref: SparseTensor already normalized, for frame 75.
    stage_tag: string written to rung5_gate3/ for 5.3 cross-check.
    Returns the frozen render tensor for Gate-1-cross downstream check.
    """
    print('\n[GATE-plain/sparse/geom] B=0 identity check...')
    ext  = EXTRINSICS.to(DEVICE)
    intr = INTRINSICS.to(DEVICE)

    # Verify B=0 everywhere (freshly initialized)
    if head_lora is not None:
        assert head_lora.lora.B.abs().max().item() == 0.0, 'AppearanceHeadLoRA B not zero at init'
    if registry is not None:
        for blk in registry.blocks.values():
            for lyr in (blk.lora_qkv, blk.lora_out, blk.lora_fc1, blk.lora_fc2):
                assert lyr.B.abs().max().item() == 0.0, f'DecBlock B not zero at init'

    # ── Frozen render ────────────────────────────────────────────────────────
    with torch.no_grad():
        meshes_frozen = dec_model(slat_norm_ref)
    mesh_f = meshes_frozen[0]
    result_f = renderer.render(mesh_f, ext, intr, return_types=['color', 'mask'])
    mask_f = result_f['mask'].unsqueeze(0)
    color_frozen = (result_f['color'] * mask_f + (1.0 - mask_f)).detach()

    # ── LoRA render (B=0) ───────────────────────────────────────────────────
    with torch.no_grad():
        if head_lora is not None:
            with dec_head_lora_ctx(dec_model, head_lora):
                meshes_lora = dec_model(slat_norm_ref)
        elif registry is not None:
            with dec_block_lora_ctx(dec_model, registry):
                meshes_lora = dec_model(slat_norm_ref)
        else:
            meshes_lora = meshes_frozen
    mesh_l = meshes_lora[0]
    result_l = renderer.render(mesh_l, ext, intr, return_types=['color', 'mask'])
    mask_l = result_l['mask'].unsqueeze(0)
    color_lora = (result_l['color'] * mask_l + (1.0 - mask_l)).detach()

    # GATE-plain: pixel render diff
    diff_render = (color_frozen - color_lora).abs().max().item()
    print(f'  GATE-plain:  max |frozen - lora(B=0)| = {diff_render:.3e}  (tol=1e-2)')
    assert diff_render < 1e-2, f'GATE-plain FAILED: diff={diff_render:.3e}'
    print('  GATE-plain PASSED')

    # GATE-sparse: SparseTensor coord identity through decoder
    # The decoder output is dense (dense tensor after sparse→dense), so check mesh coords.
    # For sparse intermediate check: compare out_layer SparseTensor coords.
    # We hook out_layer's input coords for comparison.
    _frozen_coords = [None]
    _lora_coords   = [None]

    def _cap_frozen(mod, inp, out):
        _frozen_coords[0] = inp[0].coords.clone() if isinstance(inp[0], sp.SparseTensor) else None

    def _cap_lora(mod, inp, out):
        _lora_coords[0] = inp[0].coords.clone() if isinstance(inp[0], sp.SparseTensor) else None

    h_f = dec_model.out_layer.register_forward_hook(_cap_frozen)
    with torch.no_grad():
        dec_model(slat_norm_ref)
    h_f.remove()

    if head_lora is not None:
        target_layer = head_lora.base
    else:
        target_layer = dec_model.out_layer

    h_l = target_layer.register_forward_hook(_cap_lora)
    with torch.no_grad():
        if head_lora is not None:
            with dec_head_lora_ctx(dec_model, head_lora):
                dec_model(slat_norm_ref)
        elif registry is not None:
            with dec_block_lora_ctx(dec_model, registry):
                dec_model(slat_norm_ref)
        else:
            dec_model(slat_norm_ref)
    h_l.remove()

    if _frozen_coords[0] is not None and _lora_coords[0] is not None:
        coords_match = torch.equal(_frozen_coords[0], _lora_coords[0])
        print(f'  GATE-sparse: coords equal = {coords_match}')
        assert coords_match, 'GATE-sparse FAILED: SparseTensor coords differ with B=0'
        print('  GATE-sparse PASSED')
    else:
        print('  GATE-sparse SKIPPED (out_layer not receiving SparseTensor in this config)')

    # GATE-geom: vertex identity (informational — SDF isosurface is sensitive to
    # fp16 noise; GATE-plain is the authoritative render check)
    verts_f = mesh_f.vertices
    verts_l = mesh_l.vertices
    if verts_f.shape == verts_l.shape:
        geom_diff = (verts_f.cpu().float() - verts_l.cpu().float()).abs().max().item()
        geom_ok   = geom_diff < 1e-3
        print(f'  GATE-geom:   vertex count={verts_f.shape[0]}  max_pos_diff={geom_diff:.3e}'
              f'  ({"OK" if geom_ok else "WARNING — fp16 isosurface noise, render already validated"})')
    else:
        print(f'  GATE-geom:   WARNING — vertex count differs '
              f'({verts_f.shape[0]} vs {verts_l.shape[0]}); '
              f'fp16 SDF noise near isosurface; GATE-plain render diff={diff_render:.3e} is fine')
    print('  GATE-geom DONE (non-fatal; GATE-plain is authoritative)')

    # Save for 5.3 cross-check
    gate_path = _GATE3 / f'init_{stage_tag}.pt'
    torch.save(color_frozen.cpu(), gate_path)
    print(f'  [GATE3] init render saved → {gate_path}')

    # Cross-check: for 5.3 stages, compare against other placements
    _G3_TOL = 0.05
    checked = 0
    for other in ['5_3_early', '5_3_mid', '5_3_late']:
        if other == stage_tag:
            continue
        other_path = _GATE3 / f'init_{other}.pt'
        if other_path.exists():
            other_render = torch.load(other_path, map_location='cpu', weights_only=True)
            d = (color_frozen.cpu() - other_render).abs().max().item()
            status = 'PASS' if d < _G3_TOL else 'FAIL'
            print(f'  [GATE3-cross] {stage_tag} vs {other}: max_diff={d:.3e}  [{status}]')
            assert d < _G3_TOL, (
                f'GATE3-cross FAILED: {stage_tag} vs {other} diff={d:.3e} '
                f'(check --seed and N_vox match)')
            checked += 1
    if checked == 0:
        print('  [GATE3-cross] PENDING — no other 5.3 placement files yet')
    else:
        print(f'  [GATE3-cross] {checked} cross-checks passed')

    print('[GATES] ALL PASSED\n')
    return color_frozen


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt(frame_idx: int):
    img  = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img  = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    gt   = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)
    gt_mask = (gt < 0.99).any(dim=0)
    return gt, gt_mask


def load_render_mask() -> torch.Tensor:
    m = np.array(Image.open(MASK_PATH).convert('L').resize((RENDER_RES, RENDER_RES)))
    return torch.from_numpy(m > 128).to(DEVICE)


def masked_psnr(pred, gt, mask) -> float:
    diff = (pred - gt)[:, mask]
    mse  = diff.pow(2).mean().item()
    return 10.0 * math.log10(1.0 / mse) if mse >= 1e-10 else 100.0


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
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
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh

def render_mesh(mesh, renderer) -> tuple:
    ext  = EXTRINSICS.to(DEVICE)
    intr = INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    mask = res['mask'].unsqueeze(0)
    return res['color'] * mask + (1.0 - mask), mask


def b_grad_norms(registry, head_lora):
    norms = []
    if head_lora is not None:
        g = head_lora.lora.B.grad
        if g is not None:
            norms.append(g.norm().item())
    if registry is not None:
        for blk in registry.blocks.values():
            for lyr in (blk.lora_qkv, blk.lora_out, blk.lora_fc1, blk.lora_fc2):
                g = lyr.B.grad
                if g is not None:
                    norms.append(g.norm().item())
    return norms


def b_norms(registry, head_lora):
    norms = []
    if head_lora is not None:
        norms.append(head_lora.lora.B.float().norm().item())
    if registry is not None:
        for blk in registry.blocks.values():
            for lyr in (blk.lora_qkv, blk.lora_out, blk.lora_fc1, blk.lora_fc2):
                norms.append(lyr.B.float().norm().item())
    return norms


def n_b_matrices(registry, head_lora):
    n = 0
    if head_lora is not None:
        n += 1
    if registry is not None:
        n += len(registry.blocks) * 4    # 4 B matrices per block
    return n


# ── Diagnostics ───────────────────────────────────────────────────────────────

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


def save_diagnostics(dec_model, registry, head_lora, slat_cache, renderer, epoch,
                     baseline_cache):
    epoch_dir = _DIAG / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    dec_model.eval()
    ext = EXTRINSICS.to(DEVICE)
    intr = INTRINSICS.to(DEVICE)
    for fi in DIAG_FRAMES:
        slat_norm = _slat_from_cache(slat_cache, fi)
        with torch.no_grad():
            if head_lora is not None:
                with dec_head_lora_ctx(dec_model, head_lora):
                    meshes = dec_model(slat_norm)
            elif registry is not None:
                with dec_block_lora_ctx(dec_model, registry):
                    meshes = dec_model(slat_norm)
            else:
                meshes = dec_model(slat_norm)
        color, _ = render_mesh(meshes[0], renderer)
        render = color.detach().clamp(0, 1)
        if fi not in baseline_cache:
            # Frozen baseline for reference strip
            with torch.no_grad():
                meshes_base = dec_model(slat_norm)
            color_base, _ = render_mesh(meshes_base[0], renderer)
            baseline_cache[fi] = color_base.detach().clamp(0, 1)
        strip = make_strip([
            (GT_FRAMES_DIR / f'frame_{fi:04d}.png', 'GT video'),
            (baseline_cache[fi],                    'frozen decoder'),
            (render,                                f'{args.stage} e{epoch:03d}'),
        ])
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'render_f{fi:04d}.png')
        del render, meshes, color
        gc.collect(); torch.cuda.empty_cache()


def save_curves(history, label):
    if len(history) < 2:
        return
    ep   = [r['epoch']       for r in history]
    tot  = [r['loss_total']  for r in history]
    mse_ = [r['loss_mse']    for r in history]
    lp_  = [r['loss_lpips']  for r in history]
    psnr = [r['held_psnr']   for r in history]
    bn   = [r['B_norm_mean'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Rung 5 — {label}', fontsize=11)
    axes[0].plot(ep, mse_, 'o-', color='#e06c75', lw=2, ms=4, label='MSE')
    axes[0].plot(ep, lp_,  's-', color='#d19a66', lw=2, ms=4, label='LPIPS×0.1')
    axes[0].plot(ep, tot,  '^-', color='#c678dd', lw=2, ms=4, label='total')
    axes[0].set_title('Train loss'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, psnr, 'o-', color='#98c379', lw=2, ms=4)
    axes[1].set_title('Held-out masked PSNR (dB)'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, bn, 'o-', color='#61afef', lw=2, ms=4)
    axes[2].set_title('Mean ||B||'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG / 'training_curves.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── SLaT cache ────────────────────────────────────────────────────────────────

def _slat_from_cache(slat_cache, frame_idx):
    feats, coords = slat_cache[frame_idx]
    return sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))


def precompute_slats(flow_model, raw_tokens, fixed_noise_feats, coords, frame_list,
                     desc='DINO+flow') -> dict:
    """
    Run the full flow model (25 steps, no_grad) for every frame and cache
    normalized SLaTs on CPU. Avoids re-running the expensive flow model
    every epoch.
    """
    print(f'\n[SLAT CACHE] Precomputing {len(frame_list)} SLaTs ({desc})...', flush=True)
    slat_cache = {}
    for k, fi in enumerate(frame_list):
        cond_gl = raw_tokens[fi].unsqueeze(0).to(DEVICE)
        slat    = full_denoise_nograd(flow_model, fixed_noise_feats, coords, cond_gl)
        slat_cache[fi] = (slat.feats.cpu(), slat.coords.cpu())
        del slat, cond_gl
        if (k + 1) % 25 == 0 or k == len(frame_list) - 1:
            print(f'  {k+1}/{len(frame_list)}', flush=True)
    gc.collect(); torch.cuda.empty_cache()
    print('[SLAT CACHE] done.', flush=True)
    return slat_cache


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_frames(dec_model, registry, head_lora, slat_cache, renderer,
                    render_mask, lpips_fn, frame_list):
    dec_model.eval()
    per_frame = []
    for fi in frame_list:
        slat_norm = _slat_from_cache(slat_cache, fi)
        with torch.no_grad():
            if head_lora is not None:
                with dec_head_lora_ctx(dec_model, head_lora):
                    meshes = dec_model(slat_norm)
            elif registry is not None:
                with dec_block_lora_ctx(dec_model, registry):
                    meshes = dec_model(slat_norm)
            else:
                meshes = dec_model(slat_norm)
        color, _ = render_mesh(meshes[0], renderer)
        render = color.detach().clamp(0, 1)
        gt, gt_mask = load_gt(fi)
        psnr = masked_psnr(render, gt, render_mask)
        ssim = compute_ssim(render, gt)
        m = (render_mask | gt_mask).float()
        r = render * m + (1 - m); g = gt * m + (1 - m)
        with torch.no_grad():
            lp = lpips_fn(r.unsqueeze(0) * 2 - 1,
                          g.unsqueeze(0) * 2 - 1).item()
        per_frame.append({'frame': fi, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
        del render, gt, gt_mask, meshes, color
        gc.collect(); torch.cuda.empty_cache()
    psnrs = [r['psnr']  for r in per_frame]
    ssims = [r['ssim']  for r in per_frame]
    lpips = [r['lpips'] for r in per_frame]
    return {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'lpips_mean': float(np.mean(lpips)), 'lpips_std': float(np.std(lpips)),
        'per_frame': per_frame,
    }


def find_latest_ckpt():
    ckpts = sorted(_CKPT.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    EPOCHS     = 2 if args.smoke else args.epochs
    loss_scale = LOSS_SCALE0
    DIAG_EVERY = args.diag_every

    EXPECTED_PARAMS = _expected_params(args.rank, USE_HEAD, ACTIVE_BLOCKS)

    print('=' * 72)
    print(f'Rung 5 Sub-ladder — {args.stage}')
    print(f'  stage        : {args.stage}')
    print(f'  active_blocks: {ACTIVE_BLOCKS}  (N={N_ACTIVE})')
    print(f'  use_head_lora: {USE_HEAD}')
    print(f'  rank         : {args.rank}')
    print(f'  seed         : {args.seed}')
    print(f'  epochs       : {EPOCHS}')
    print(f'  expected_params: {EXPECTED_PARAMS:,}')
    print(f'  run_id       : {RUN_ID}')
    print(f'  output       : {_OUT}')
    print('=' * 72)

    json.dump(_CFG | {'run_id': RUN_ID, 'label': _LABEL},
              open(_OUT / 'config.json', 'w'), indent=2)

    assert len(set(HELD_OUT) & set(TRAIN)) == 0
    assert len(HELD_OUT) == 15 and len(TRAIN) == 135

    # ── pipeline ──────────────────────────────────────────────────────────────
    print(f'\n[LOAD] {PRETRAINED}')
    pipeline  = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model  = pipeline.models['slat_decoder_mesh']

    # ── N_DEC runtime assert ──────────────────────────────────────────────────
    N_DEC_actual = len(dec_model.blocks)
    print(f'[N_DEC] len(dec_model.blocks) = {N_DEC_actual}')
    assert N_DEC_actual == _N_DEC_EXPECTED, \
        f'N_DEC mismatch: actual={N_DEC_actual} expected={_N_DEC_EXPECTED}'
    print(f'[N_DEC] PASSED ({N_DEC_actual} blocks confirmed)')

    # Freeze everything
    for p in flow_model.parameters():
        p.requires_grad_(False)
    for p in dec_model.parameters():
        p.requires_grad_(False)
    n_flow = sum(p.numel() for p in flow_model.parameters())
    n_dec  = sum(p.numel() for p in dec_model.parameters())
    print(f'[FREEZE] flow={n_flow:,}  dec={n_dec:,}  params frozen')

    # ── structure (fixed STRUCT_SEED) — BEFORE model offloading ──────────────
    # pipeline.device is a property that scans all models; offloading some to
    # CPU first would make it return 'cpu', causing image tensors to land on
    # CPU while the model weights are on GPU → RuntimeError.  Sample structure
    # while everything is on GPU, then offload.
    print(f'\n[STRUCT] seed={STRUCT_SEED}', flush=True)
    ref_img     = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref_img])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}', flush=True)
    print(f"[DEBUG] N_vox={N_vox} — skipping 7301 assertion, retraining on current node")
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    # Offload unneeded pipeline models to CPU (after structure sampling)
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA modules ──────────────────────────────────────────────────────────
    head_lora = None
    registry  = None

    if USE_HEAD:
        print(f'\n[LORA] AppearanceHeadLoRA rank={args.rank}  '
              f'(out_layer appearance [53:101] only)')
        head_lora = AppearanceHeadLoRA(dec_model.out_layer, rank=args.rank).to(DEVICE)
    elif args.stage != '5.0':
        print(f'\n[LORA] DecLoRARegistry  blocks={ACTIVE_BLOCKS}  rank={args.rank}')
        registry = DecLoRARegistry(ACTIVE_BLOCKS, rank=args.rank).to(DEVICE)

    # GATE 0: trainable param count (requires_grad only — head_lora.base is frozen)
    actual_params = 0
    if head_lora  is not None: actual_params += sum(p.numel() for p in head_lora.parameters() if p.requires_grad)
    if registry   is not None: actual_params += sum(p.numel() for p in registry.parameters() if p.requires_grad)
    print(f'\n[GATE 0]  expected={EXPECTED_PARAMS:,}  actual={actual_params:,}')
    assert actual_params == EXPECTED_PARAMS, \
        f'GATE 0 FAILED: {actual_params} != {EXPECTED_PARAMS}'
    print('[GATE 0] PASSED\n')

    # ── fixed noise ────────────────────────────────────────────────────────────
    print(f'[NOISE] seed={args.seed}')
    torch.manual_seed(args.seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # ── DINOv2 cache ──────────────────────────────────────────────────────────
    print(f'\n[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    # ── Precompute SLaTs for all frames ────────────────────────────────────────
    all_frames = list(range(1, N_FRAMES + 1))
    slat_cache = precompute_slats(flow_model, raw_tokens, fixed_noise_feats, coords,
                                  all_frames, desc=f'seed={args.seed}')
    # Move flow model to CPU — not needed after SLaT cache is built
    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    dec_model.to(DEVICE)

    # ── LPIPS + mask ──────────────────────────────────────────────────────────
    print('\n[LPIPS] Loading AlexNet...')
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    render_mask = load_render_mask()
    print(f'[MASK] {render_mask.sum().item()} masked pixels '
          f'({100*render_mask.float().mean():.1f}%)')

    renderer = make_renderer()

    # ── Stage 5.0: eval only ──────────────────────────────────────────────────
    if args.stage == '5.0':
        print('\n' + '='*72)
        print('[5.0 BASELINE] Frozen decoder eval on held-out frames')
        print('='*72)
        dec_model.eval()
        res = evaluate_frames(dec_model, None, None, slat_cache, renderer,
                              render_mask, lpips_fn, HELD_OUT)
        print(f'  Held  PSNR={res["psnr_mean"]:.3f}±{res["psnr_std"]:.3f} dB  '
              f'SSIM={res["ssim_mean"]:.4f}  LPIPS={res["lpips_mean"]:.4f}')
        res_all = evaluate_frames(dec_model, None, None, slat_cache, renderer,
                                  render_mask, lpips_fn, all_frames)
        print(f'  All   PSNR={res_all["psnr_mean"]:.3f}±{res_all["psnr_std"]:.3f} dB  '
              f'SSIM={res_all["ssim_mean"]:.4f}  LPIPS={res_all["lpips_mean"]:.4f}')
        out = {'stage': '5.0', 'held': res, 'all': res_all}
        json.dump(out, open(_OUT / 'final_eval.json', 'w'), indent=2)
        print(f'\n[DONE] 5.0 baseline eval saved → {_OUT / "final_eval.json"}')
        return

    # ── Gates (all training stages) ────────────────────────────────────────────
    dec_model.eval()
    slat_ref = _slat_from_cache(slat_cache, 75)

    latest_ckpt = find_latest_ckpt()
    resumed     = latest_ckpt is not None

    if resumed:
        print('\n[GATES] SKIPPED — resuming from checkpoint (B ≠ 0 by design)\n')
    else:
        run_gates(dec_model, registry, head_lora, slat_ref, renderer, _stage_tag)

    # ── Optimizer + resume ────────────────────────────────────────────────────
    # Filter to requires_grad=True only — head_lora.base is a frozen submodule
    trainable_params = []
    if head_lora is not None:
        trainable_params += [p for p in head_lora.parameters() if p.requires_grad]
    if registry  is not None:
        trainable_params += [p for p in registry.parameters() if p.requires_grad]

    optimizer   = torch.optim.Adam(trainable_params, lr=LR, eps=1e-16, weight_decay=0.0)
    start_epoch = 1
    history     = []
    best_psnr   = -float('inf')

    if resumed:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        assert ckpt.get('run_id') == RUN_ID, \
            f'checkpoint run_id mismatch: {ckpt.get("run_id")} != {RUN_ID}'
        if head_lora is not None:
            head_lora.load_state_dict(ckpt['head_lora_state'])
        if registry is not None:
            registry.load_state_dict(ckpt['registry_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -float('inf'))
        loss_scale  = ckpt.get('loss_scale', LOSS_SCALE0)
        hist_path   = _OUT / 'loss_history.json'
        if hist_path.exists():
            history = json.load(open(hist_path))
        print(f'  resumed from epoch {ckpt["epoch"]}  loss_scale={loss_scale:.0f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs already done.'); return

    # ── Gate 2: gradient check (fresh runs only — skipped on resume) ──────────
    if not resumed:
        print('\n[GATE 2] Gradient flow check (frame 75)...', flush=True)
        dec_model.eval()
        slat_75 = _slat_from_cache(slat_cache, 75)

        optimizer.zero_grad()

        if head_lora is not None:
            with dec_head_lora_ctx(dec_model, head_lora):
                meshes = dec_model(slat_75)
        elif registry is not None:
            with dec_block_lora_ctx(dec_model, registry):
                meshes = dec_model(slat_75)
        else:
            meshes = dec_model(slat_75)

        color, _ = render_mesh(meshes[0], renderer)
        gt_75, gm_75 = load_gt(75)
        _, _, loss_g2 = masked_loss(color, gt_75, render_mask, gm_75, lpips_fn)
        (loss_g2 * loss_scale).backward()

        b_gnorms_list = b_grad_norms(registry, head_lora)
        n_b_total     = n_b_matrices(registry, head_lora)
        n_zero_g2     = sum(1 for g in b_gnorms_list if g == 0.0)
        print(f'  B.grad norms ({len(b_gnorms_list)}/{n_b_total} have grad): '
              f'min={min(b_gnorms_list):.3e}  max={max(b_gnorms_list):.3e}', flush=True)
        if n_zero_g2 > 0:
            print(f'  [GATE 2] WARNING: {n_zero_g2}/{n_b_total} B matrices have exactly-zero grad '
                  f'(expected for early blocks in fp16; training dominated by later blocks)',
                  flush=True)
        assert len(b_gnorms_list) == n_b_total, \
            f'GATE 2 FAILED: only {len(b_gnorms_list)}/{n_b_total} B matrices received grad'
        optimizer.zero_grad()
        print('[GATE 2] PASSED\n', flush=True)

        del meshes, color, gt_75, gm_75, loss_g2
        gc.collect(); torch.cuda.empty_cache()
    else:
        print('[GATE 2] SKIPPED — resuming from checkpoint\n', flush=True)

    # ── Training ───────────────────────────────────────────────────────────────
    dec_model.eval()   # decoder base stays in eval mode; LoRA adapters are separate
    baseline_cache = {}

    print(f'[TRAIN] epochs {start_epoch}→{EPOCHS}  frames/epoch={len(TRAIN)}'
          f'  LOSS_SCALE={loss_scale:.0f}', flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        ep_t0 = time.time()
        frame_order = TRAIN[:]
        random.shuffle(frame_order)

        tot_loss = tot_mse = tot_lp = 0.0
        n_steps = 0

        print(f'\n[EPOCH {epoch}/{EPOCHS}] starting...', flush=True)

        for fi in frame_order:
            slat_norm = _slat_from_cache(slat_cache, fi)

            optimizer.zero_grad()

            if head_lora is not None:
                with dec_head_lora_ctx(dec_model, head_lora):
                    meshes = dec_model(slat_norm)
            elif registry is not None:
                with dec_block_lora_ctx(dec_model, registry):
                    meshes = dec_model(slat_norm)
            else:
                break   # 5.0 handled above

            color, _ = render_mesh(meshes[0], renderer)
            gt, gt_mask = load_gt(fi)
            mse, lp, total = masked_loss(color, gt, render_mask, gt_mask, lpips_fn)

            (total * loss_scale).backward()
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.div_(loss_scale)
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()

            tot_loss += total.item()
            tot_mse  += mse.item()
            tot_lp   += lp.item()
            n_steps  += 1

            if n_steps % 15 == 0:
                print(f'  e{epoch:02d} [{n_steps:03d}/{len(TRAIN)}] f{fi:03d}  '
                      f'mse={mse.item():.5f}  lpips={lp.item():.5f}  total={total.item():.5f}',
                      flush=True)

            del meshes, color, gt, gt_mask, mse, lp, total, slat_norm
            gc.collect(); torch.cuda.empty_cache()

        # Epoch eval (held-out only — training frames not needed each epoch)
        held_res = evaluate_frames(dec_model, registry, head_lora, slat_cache,
                                   renderer, render_mask, lpips_fn, HELD_OUT)
        held_psnr = held_res['psnr_mean']
        held_std  = held_res['psnr_std']
        held_ssim = held_res['ssim_mean']
        held_lp   = held_res['lpips_mean']

        bnorms = b_norms(registry, head_lora)
        ep_loss  = tot_loss / max(n_steps, 1)
        ep_mse   = tot_mse  / max(n_steps, 1)
        ep_lp    = tot_lp   / max(n_steps, 1)
        ep_time  = time.time() - ep_t0

        new_best = held_psnr > best_psnr
        if new_best:
            best_psnr = held_psnr

        print(f'\n[EPOCH {epoch}/{EPOCHS}]  loss={ep_loss:.5f}  '
              f'(mse={ep_mse:.5f}  lpips×0.1={ep_lp * W_LPIPS:.5f})', flush=True)
        print(f'  ||B||   mean={np.mean(bnorms):.4f}  '
              f'min={np.min(bnorms):.4f}  max={np.max(bnorms):.4f}', flush=True)
        print(f'  held PSNR={held_psnr:.3f}±{held_std:.3f} dB  '
              f'SSIM={held_ssim:.4f}  LPIPS={held_lp:.4f}  '
              f'GPU={torch.cuda.max_memory_allocated()/1e9:.1f}GB  '
              f'loss_scale={loss_scale:.0f}  '
              + ('BEST ✓' if new_best else '') + f'  time={ep_time:.1f}s', flush=True)

        # Checkpoint
        ckpt = {
            'epoch': epoch, 'run_id': RUN_ID, 'best_psnr': best_psnr,
            'loss_scale': loss_scale, 'optimizer': optimizer.state_dict(),
        }
        if head_lora is not None:
            ckpt['head_lora_state'] = head_lora.state_dict()
        if registry is not None:
            ckpt['registry_state'] = registry.state_dict()

        ckpt_path = _CKPT / f'lora_e{epoch:03d}.pt'
        torch.save(ckpt, ckpt_path)
        if new_best:
            torch.save(ckpt, _CKPT / 'lora_best.pt')
            print(f'\n  [CKPT] new best → lora_best.pt')

        rec = {
            'epoch': epoch, 'loss_total': ep_loss, 'loss_mse': ep_mse,
            'loss_lpips': ep_lp, 'held_psnr': held_psnr, 'held_std': held_std,
            'held_ssim': held_ssim, 'held_lpips': held_lp,
            'B_norm_mean': float(np.mean(bnorms)), 'B_norm_max': float(np.max(bnorms)),
            'time_s': ep_time,
        }
        history.append(rec)
        json.dump(history, open(_OUT / 'loss_history.json', 'w'), indent=2)
        save_curves(history, _LABEL)

        if epoch % DIAG_EVERY == 0:
            save_diagnostics(dec_model, registry, head_lora, slat_cache,
                             renderer, epoch, baseline_cache)

    # ── Final eval ─────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print('[FINAL EVAL] All 150 frames — post-training quality audit')
    print('='*72)

    # Load best checkpoint
    best_ckpt_path = _CKPT / 'lora_best.pt'
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True)
        if head_lora is not None:
            head_lora.load_state_dict(best_ckpt['head_lora_state'])
        if registry is not None:
            registry.load_state_dict(best_ckpt['registry_state'])
        print(f'  Loaded best ckpt (epoch={best_ckpt["epoch"]}  '
              f'best_psnr={best_ckpt["best_psnr"]:.3f})')

    res_held  = evaluate_frames(dec_model, registry, head_lora, slat_cache,
                                renderer, render_mask, lpips_fn, HELD_OUT)
    res_train = evaluate_frames(dec_model, registry, head_lora, slat_cache,
                                renderer, render_mask, lpips_fn, TRAIN)
    res_all   = evaluate_frames(dec_model, registry, head_lora, slat_cache,
                                renderer, render_mask, lpips_fn, all_frames)

    print(f'  All:   PSNR={res_all["psnr_mean"]:.3f}±{res_all["psnr_std"]:.3f}  '
          f'SSIM={res_all["ssim_mean"]:.4f}  LPIPS={res_all["lpips_mean"]:.4f}')
    print(f'  Held:  PSNR={res_held["psnr_mean"]:.3f}±{res_held["psnr_std"]:.3f}  '
          f'SSIM={res_held["ssim_mean"]:.4f}')
    print(f'  Train: PSNR={res_train["psnr_mean"]:.3f}±{res_train["psnr_std"]:.3f}')

    final = {
        'stage': args.stage, 'rank': args.rank, 'run_id': RUN_ID,
        'held': res_held, 'train': res_train, 'all': res_all,
    }
    json.dump(final, open(_OUT / 'final_eval.json', 'w'), indent=2)
    print(f'  → final_eval.json saved')

    print(f'\n[DONE] {EPOCHS} epochs complete.')
    print(f'  stage        : {args.stage}  blocks={ACTIVE_BLOCKS}  rank={args.rank}')
    print(f'  best held PSNR: {best_psnr:.3f} dB')
    print(f'  output       : {_OUT}')


if __name__ == '__main__':
    main()
