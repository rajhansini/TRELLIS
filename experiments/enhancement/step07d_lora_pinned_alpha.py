"""
Step 07d — LoRA training with pinned alpha and fixed LoRA scaling.

Fixes vs step07c:
  FIX 1: A init kaiming_uniform, scaling=(lora_alpha/rank)=1.0  (via lora_v2_fixed)
  FIX 2: alpha is a fixed float hyperparameter, NOT nn.Parameter
  FIX 3: no ALPHA_REG (unnecessary when alpha is not learnable)

Alpha sweep: run separately at --alpha 0.1, 0.25, 0.5
Results dir: results_mcfm_{mode}_d_alpha{alpha_str}_seed6/

BACKWARD COMPATIBLE: step07c, lora_v2.py, all existing results untouched.

Usage:
  python step07d_lora_pinned_alpha.py --mode v2_C --alpha 0.25 --epochs 15
  python step07d_lora_pinned_alpha.py --mode v2_C --alpha 0.1  --epochs 15
  python step07d_lora_pinned_alpha.py --mode v2_C --alpha 0.5  --epochs 15
  python step07d_lora_pinned_alpha.py --mode v2_C --alpha 0.25 --smoke
"""

import sys, os, argparse as _ap
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode',    type=str,   default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_pre.add_argument('--alpha',   type=float, default=0.25,
                  help='Fixed LoRA gating scalar (not learnable). Sweep: 0.1 / 0.25 / 0.5')
_pre.add_argument('--epochs',  type=int,   default=15)
_pre.add_argument('--rank',    type=int,   default=4)
_pre.add_argument('--lr',      type=float, default=1e-4)
_pre.add_argument('--loss_scale', type=float, default=4096.0)
_pre.add_argument('--smoke',   action='store_true')
_PRE, _ = _pre.parse_known_args()

if _PRE.smoke:
    _PRE.epochs = 1

_alpha_str = f'{_PRE.alpha:.2f}'.replace('.', 'p')
_lr_tag    = ('' if abs(_PRE.lr - 1e-4) < 1e-10
              else '_lr' + f'{_PRE.lr:.0e}'.replace('-0', 'm').replace('+0', ''))
_RESULTS   = _HERE / f'results_mcfm_{_PRE.mode}_d_alpha{_alpha_str}{_lr_tag}_seed6'
_CKPT_DIR  = _RESULTS / 'lora_ckpts'
_DIAG_DIR  = _RESULTS / 'diag_renders'
_RESULTS.mkdir(parents=True, exist_ok=True)
_CKPT_DIR.mkdir(exist_ok=True)
_DIAG_DIR.mkdir(exist_ok=True)
_LOG_NAME  = (f'smoke_{_PRE.mode}_alpha{_alpha_str}.log' if _PRE.smoke
              else f'train_{_PRE.mode}_alpha{_alpha_str}_epochs{_PRE.epochs}.log')


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'a', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(_RESULTS / _LOG_NAME)
sys.stderr = sys.stdout

# ── nvdiffrast arch guard ─────────────────────────────────────────────────────
# Installs nvdiffrast into node-local /tmp (not shared NFS) so concurrent
# builds on different-arch nodes can't overwrite each other.
def _ensure_nvdiffrast():
    import subprocess, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm{p.major}{p.minor}'
    arch_str = f'{p.major}.{p.minor}'
    local    = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)

    # Prepend node-local dir so it wins over shared conda env
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)

    try:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext()
        del glctx
        print(f'[NVDIFF] OK', flush=True)
        return
    except Exception as e:
        print(f'[NVDIFF] FAILED: {e}', flush=True)

    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')

    print(f'[NVDIFF] building for {arch_tag} -> {local} (~5 min)', flush=True)
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    print(f'[NVDIFF] build done — restarting to load new binary', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()
# ─────────────────────────────────────────────────────────────────────────────

import json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

from step4_mcfm.mcfm           import mcfm_v2, mcfm_v3
from step6_5_lora.lora_v2_fixed import (build_lora_blocks, freeze_trellis,
                                         gate0_verify, trainable_params,
                                         count_trainable)
from step6_5_lora.dual_path_v2  import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render, _gfn,
                                               RENDER_RES, SLAT_MEAN, SLAT_STD)

# ── Constants ──────────────────────────────────────────────────────────────────
VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
GT_FRAMES_DIR  = VIDEO_FRAMES_DIR
GT_FRAME_75    = VIDEO_FRAMES_DIR / 'frame_0075.png'
PRETRAINED     = 'JeffreyXiang/TRELLIS-image-large'
DEVICE         = torch.device('cuda')
N_FRAMES       = 150
STRUCT_SEED    = 42
FIXED_SEED     = 6
STEPS          = 25
RESCALE_T      = 3.0
GRAD_CLIP      = 1.0

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DIAG_KEYFRAMES = [1, 75, 150]


def get_window(frame_idx, mode):
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def encode_frame(dino_model, frame_idx):
    img  = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def load_gt_tensor(frame_idx):
    img = Image.open(GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1).to(DEVICE)


def blend(mode, raw_tokens, frame_idx):
    win     = get_window(frame_idx, mode)
    lam     = torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)
    tok_d   = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    stacked  = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0)
    weights  = torch.tensor([2.0 if i == frame_idx else 1.0 for i in win],
                             dtype=torch.float32, device=DEVICE)
    weights  /= weights.sum()
    K_pooled = (weights[:, None, None] * stacked).sum(0)
    return K_hat, K_pooled, win


def denoise_prefix_nograd(flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, alpha):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def denoise_last_step(flow_model, x_in, cond_gl, K_pooled, lora_blocks, alpha, require_grad):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
    ctx   = dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None)
    if require_grad:
        with ctx:
            v = flow_model(x_in, t_ten, cond_gl)
    else:
        with torch.no_grad():
            with ctx:
                v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)


def render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                        raw_tokens, mode, frame_idx, coords, fixed_noise_feats, renderer):
    K_hat, K_pooled, _ = blend(mode, raw_tokens, frame_idx)
    ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl = K_hat.unsqueeze(0)
    flow_model.to(DEVICE); gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled.to(DEVICE), lora_blocks,
                                   alpha, enhance_bias=None):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    slat = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    flow_model.to(DEVICE)
    return color.detach().clamp(0, 1)


def lora_b_norm(lora_blocks):
    vals = [p.float().norm().item()
            for blk in lora_blocks
            for name, p in blk.named_parameters() if 'B' in name]
    return float(np.sqrt(np.mean([v**2 for v in vals]))) if vals else 0.0


def lora_ba_norm(lora_blocks):
    """Mean ||B@A|| across all blocks (both q and kv)."""
    vals = []
    for blk in lora_blocks:
        vals.append((blk.lora_q.B.float()  @ blk.lora_q.A.float()).norm().item())
        vals.append((blk.lora_kv.B.float() @ blk.lora_kv.A.float()).norm().item())
    return float(np.mean(vals))


def make_strip(panels_info, cell=320, label_h=28, font=None):
    n = len(panels_info)
    canvas = Image.new('RGB', (cell * n, cell + label_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (img_or_path, lbl) in enumerate(panels_info):
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert('RGB').resize((cell, cell), Image.LANCZOS)
        else:
            arr = (img_or_path.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((cell, cell), Image.LANCZOS)
        canvas.paste(img, (col * cell, label_h))
        draw.rectangle([col*cell, 0, (col+1)*cell-1, label_h-1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0, 0), lbl, font=font)[2]
        except: tw = len(lbl) * 7
        draw.text((col*cell + (cell-tw)//2, 6), lbl, fill=(210, 210, 210), font=font)
    return canvas


def save_epoch_diagnostics(pipeline, flow_model, lora_blocks, alpha,
                            raw_tokens, mode, coords, fixed_noise_feats,
                            renderer, epoch, mcfm_cache, font):
    epoch_dir = _DIAG_DIR / f'e{epoch:03d}'
    epoch_dir.mkdir(exist_ok=True)
    for fi in DIAG_KEYFRAMES:
        print(f'  [DIAG] f{fi}...')
        lora_t = render_frame_nograd(flow_model, pipeline, lora_blocks, alpha,
                                     raw_tokens, mode, fi, coords, fixed_noise_feats, renderer)
        if fi not in mcfm_cache:
            # alpha=0 render as control
            mcfm_cache[fi] = render_frame_nograd(flow_model, pipeline, lora_blocks, 0.0,
                                                  raw_tokens, mode, fi, coords,
                                                  fixed_noise_feats, renderer)
        gt_path = GT_FRAMES_DIR / f'frame_{fi:04d}.png'
        strip   = make_strip([
            (gt_path,          'GT video'),
            (mcfm_cache[fi],   'alpha=0 control'),
            (lora_t,           f'LoRA alpha={alpha:.2f} e{epoch:03d}'),
        ], font=font)
        strip.save(epoch_dir / f'strip_f{fi:04d}.png')
        arr = (lora_t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(epoch_dir / f'lora_f{fi:04d}.png')
        # pixel stats
        print(f'    f{fi}: lora mean={lora_t.mean():.4f} std={lora_t.std():.4f}  '
              f'ctrl mean={mcfm_cache[fi].mean():.4f} std={mcfm_cache[fi].std():.4f}')
        del lora_t
        gc.collect(); torch.cuda.empty_cache()


def save_loss_curve(history):
    if not history:
        return
    ep  = [r['epoch']         for r in history]
    tl  = [r['avg_task_loss'] for r in history]
    bn  = [r['lora_B_norm']   for r in history]
    ban = [r['lora_BA_norm']  for r in history]
    eff = [r['eff_delta']     for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'step07d {_PRE.mode} alpha={_PRE.alpha:.2f}', fontsize=11)
    axes[0].plot(ep, tl, 'o-', color='#e06c75', lw=2, ms=5)
    axes[0].set_title('Task loss'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, bn,  'o-', color='#61afef', lw=2, ms=5, label='||B||')
    axes[1].plot(ep, ban, 's-', color='#98c379', lw=2, ms=5, label='||B@A||')
    axes[1].set_title('Weight norms'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, eff, 'o-', color='#c678dd', lw=2, ms=5)
    axes[2].set_title(f'alpha×||B@A||  (effective delta, alpha={_PRE.alpha:.2f})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_DIAG_DIR / 'loss_curve.png', dpi=130, bbox_inches='tight')
    plt.close()


def find_latest_ckpt():
    ckpts = sorted(_CKPT_DIR.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None


def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode',       type=str,   default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--alpha',      type=float, default=0.25)
    parser.add_argument('--epochs',     type=int,   default=15)
    parser.add_argument('--rank',       type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--loss_scale', type=float, default=4096.0)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--diag_every', type=int,   default=1)
    parser.add_argument('--smoke',      action='store_true')
    args = parser.parse_args()

    mode         = args.mode
    ALPHA        = float(args.alpha)      # fixed scalar, not nn.Parameter
    EPOCHS       = args.epochs
    LORA_RANK    = args.rank
    LR           = args.lr
    LOSS_SCALE   = args.loss_scale
    FRAME_STRIDE = args.frame_stride
    DIAG_EVERY   = args.diag_every
    SMOKE        = args.smoke

    print('=' * 72)
    print('Step 07d — LoRA training, pinned alpha, fixed scaling')
    print('=' * 72)
    print(f'  mode        : {mode}')
    print(f'  alpha       : {ALPHA}  (FIXED — not learnable)')
    print(f'  epochs      : {EPOCHS}')
    print(f'  rank        : {LORA_RANK}  (lora_alpha=rank → scaling=1.0)')
    print(f'  lr          : {LR}')
    print(f'  loss_scale  : {LOSS_SCALE}')
    print(f'  results     : {_RESULTS}')

    # ── GT check ──────────────────────────────────────────────────────────────
    missing = [i for i in [1, 75, 150]
               if not (GT_FRAMES_DIR / f'frame_{i:04d}.png').exists()]
    if missing:
        raise FileNotFoundError(f'GT frames missing: {missing}')

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = ImageFont.load_default()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print('\n[LOAD] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Structure ─────────────────────────────────────────────────────────────
    print(f'\n[STRUCT] STRUCT_SEED={STRUCT_SEED}...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}')
    assert N_vox == 7301, f'N_vox={N_vox} != 7301'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print('\n[LORA] Building LoRA blocks (lora_v2_fixed)...')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, lora_alpha=LORA_RANK).to(DEVICE)
    gate0_verify(lora_blocks)
    print(f'  trainable params: {count_trainable(lora_blocks):,}')

    # ── Optimizer + resume ────────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(trainable_params(lora_blocks), lr=LR,
                                    eps=1e-16, weight_decay=0.0)
    loss_scale   = LOSS_SCALE   # mutable: halved on overflow
    start_epoch  = 1
    loss_history = []
    best_task_loss = float('inf')
    latest_ckpt  = find_latest_ckpt()

    if latest_ckpt:
        print(f'\n[RESUME] {latest_ckpt.name}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        assert ckpt.get('script') == 'step07d', \
            f'Checkpoint from wrong script: {ckpt.get("script")}'
        assert abs(ckpt.get('alpha', -1) - ALPHA) < 1e-6, \
            f'Checkpoint alpha={ckpt.get("alpha")} != --alpha={ALPHA}'
        lora_blocks.load_state_dict(ckpt['lora_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch    = ckpt['epoch'] + 1
        best_task_loss = ckpt.get('best_task_loss', float('inf'))
        hist_path = _RESULTS / 'loss_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                loss_history = json.load(f)
        print(f'  resumed from epoch {ckpt["epoch"]}  task={ckpt["avg_task_loss"]:.5f}')
    else:
        print('\n[RESUME] No checkpoint — starting fresh.')

    if start_epoch > EPOCHS:
        print(f'All {EPOCHS} epochs done.'); return

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] FIXED_SEED={FIXED_SEED}')
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # ── DINOv2 ────────────────────────────────────────────────────────────────
    print(f'\n[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); torch.cuda.empty_cache()

    renderer = make_renderer()
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.train()

    frame_list = [1, 75, 150] if SMOKE else list(range(1, N_FRAMES + 1, FRAME_STRIDE))
    print(f'\n[TRAIN] epochs {start_epoch}→{EPOCHS}  frames/epoch={len(frame_list)}  '
          f'alpha={ALPHA}(fixed)  LOSS_SCALE={LOSS_SCALE}  eps=1e-16  weight_decay=0')

    first_diag = True
    mcfm_cache = {}

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_task_loss = 0.0
        t_epoch = time.time()

        for frame_i in frame_list:
            K_hat, K_pooled, win = blend(mode, raw_tokens, frame_i)
            cond_gl  = K_hat.unsqueeze(0)
            noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
            optimizer.zero_grad()

            x_prefix = denoise_prefix_nograd(
                flow_model, noise_sp, cond_gl, K_pooled, lora_blocks, ALPHA)

            x0_val   = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, ALPHA,
                require_grad=False)
            slat_val = normalize_slat(x0_val)
            del x0_val, noise_sp

            _flow_ref = pipeline.models.get('slat_flow_model')
            if _flow_ref is not None: _flow_ref.cpu()
            gc.collect(); torch.cuda.empty_cache()

            try:
                color, slat_leaf_feats = decode_and_render(
                    pipeline, slat_val, renderer, diag=first_diag, device=DEVICE)
                gc.collect(); torch.cuda.empty_cache()

                gt        = load_gt_tensor(frame_i)
                task_loss = F.mse_loss(color, gt)

                if first_diag:
                    print(f'  [G4] task_loss={_gfn(task_loss)}  val={task_loss.item():.5f}')

                (task_loss * loss_scale).backward()
                # Do NOT divide yet — keep loss_scale active through flow model backward

                if first_diag:
                    g = slat_leaf_feats.grad
                    print(f'  [G5] slat_leaf_feats.grad (×{loss_scale:.0f}): '
                          f'{"None" if g is None else f"dtype={g.dtype} max={g.abs().max():.3e}"}')

            finally:
                if _flow_ref is not None: _flow_ref.to(DEVICE)

            if slat_leaf_feats.grad is None:
                raise RuntimeError('decode/render: no gradient on slat_leaf_feats')

            # grad_slat still at loss_scale amplitude — keeps signal alive through frozen blocks
            grad_slat = slat_leaf_feats.grad.detach()
            x0_raw    = denoise_last_step(
                flow_model, x_prefix, cond_gl, K_pooled, lora_blocks, ALPHA,
                require_grad=True)
            slat = normalize_slat(x0_raw)
            torch.autograd.backward(slat.feats, grad_slat)

            # Unscale LoRA grads now, before optimizer
            lora_params = trainable_params(lora_blocks)
            for p in lora_params:
                if p.grad is not None:
                    p.grad.div_(loss_scale)

            # Gate: print B.grad norms on first step to confirm signal arrived
            if first_diag:
                b_grads = [p.grad.norm().item()
                           for blk in lora_blocks
                           for name, p in blk.named_parameters()
                           if 'B' in name and p.grad is not None]
                print(f'  [GATE] B.grad norms across 48 B matrices: '
                      f'min={min(b_grads):.3e}  max={max(b_grads):.3e}  '
                      f'nonzero={sum(v > 0 for v in b_grads)}/48')
                first_diag = False

            # inf/nan guard: skip step and halve loss_scale if overflow
            has_bad = any(not torch.isfinite(p.grad).all()
                          for p in lora_params if p.grad is not None)
            if has_bad:
                print(f'  [SCALE] inf/nan in grads — skipping step, loss_scale {loss_scale:.0f} → {loss_scale/2:.0f}')
                optimizer.zero_grad()
                loss_scale /= 2.0
                del color, slat_leaf_feats, gt, task_loss
                del x0_raw, slat, grad_slat, x_prefix, slat_val, cond_gl, K_pooled, K_hat
                gc.collect(); torch.cuda.empty_cache()
                continue

            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step()

            task_loss_val    = task_loss.item()
            epoch_task_loss += task_loss_val

            if frame_i == frame_list[0] or frame_i % 10 == 0:
                print(f'  e{epoch:02d} f{frame_i:03d}  task={task_loss_val:.5f}  win={win}')

            del color, slat_leaf_feats, gt, task_loss
            del x0_raw, slat, grad_slat, x_prefix, slat_val, cond_gl, K_pooled, K_hat
            gc.collect(); torch.cuda.empty_cache()

        # ── Epoch summary ──────────────────────────────────────────────────────
        avg_task = epoch_task_loss / len(frame_list)
        b_norm   = lora_b_norm(lora_blocks)
        ba_norm  = lora_ba_norm(lora_blocks)
        eff      = ALPHA * ba_norm
        elapsed  = time.time() - t_epoch
        is_best  = avg_task < best_task_loss
        if is_best:
            best_task_loss = avg_task

        print(f'[EPOCH {epoch:02d}] avg_task={avg_task:.5f}  '
              f'||B||={b_norm:.4f}  ||B@A||={ba_norm:.4f}  '
              f'eff_delta={eff:.4f}  loss_scale={loss_scale:.0f}  '
              f'{"BEST ✓" if is_best else ""}  time={elapsed:.1f}s')

        loss_history.append({
            'epoch': epoch, 'avg_task_loss': avg_task,
            'lora_B_norm': b_norm, 'lora_BA_norm': ba_norm,
            'eff_delta': eff, 'is_best': is_best,
        })
        with open(_RESULTS / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

        if not SMOKE:
            ckpt_data = {
                'script'        : 'step07d',
                'epoch'         : epoch,
                'mode'          : mode,
                'alpha'         : ALPHA,
                'rank'          : LORA_RANK,
                'lora_state'    : lora_blocks.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'avg_task_loss' : avg_task,
                'best_task_loss': best_task_loss,
            }
            torch.save(ckpt_data, _CKPT_DIR / f'lora_e{epoch:03d}.pt')
            if is_best:
                torch.save(ckpt_data, _CKPT_DIR / 'lora_best.pt')
                print(f'  [CKPT] new best → lora_best.pt')

        do_diag = (epoch % DIAG_EVERY == 0 or epoch == EPOCHS) and not SMOKE
        if do_diag:
            print(f'  [DIAG] epoch {epoch} visual diagnostics...')
            flow_model.eval()
            save_epoch_diagnostics(pipeline, flow_model, lora_blocks, ALPHA,
                                   raw_tokens, mode, coords, fixed_noise_feats,
                                   renderer, epoch, mcfm_cache, font)
            save_loss_curve(loss_history)
            flow_model.train()
            gc.collect(); torch.cuda.empty_cache()

        gc.collect(); torch.cuda.empty_cache()

    print(f'\n[DONE] {EPOCHS} epochs. best_task_loss={best_task_loss:.5f}')
    print(f'  Results: {_RESULTS}')


if __name__ == '__main__':
    main()
