"""
LoRA Checkpoint Diagnostics — 6 tests on the latest v2_C realgt run (epoch 18).

Diagnostic 1 : Force alpha=0 on trained checkpoint → render f001/f075/f150.
                 Good (same as MCFM ref) → LoRA is strictly harmful, confirmed.
                 Bad → bug is upstream of LoRA.

Diagnostic 2 : Print alpha from checkpoint.
                 Diverged well above 0.5 → unbounded growth is the cause.

Diagnostic 3 : (K_pooled[1] - K_pooled[150]).abs().mean()
                 vs (raw_tokens[1] - raw_tokens[150]).abs().mean()
                 First tiny, second not → PATH B is blind to frame identity.

Diagnostic 4 : (K_hat[t] - raw_tokens[t]).abs().mean() across all 150 frames.
                 Near zero → blending is a no-op.

Diagnostic 5 : Plot training loss curve (task + total, alpha, lora_B_norm).

Diagnostic 6 : Per-block LoRA weight norms from checkpoint.
                 Exploding → runaway.  Near-zero with bad render → alpha damage.

All outputs → lora_diagnostic/  (same dir as this script).
Log → lora_diagnostic/diag_lora_checkpoint.log
"""

import sys, os, gc, time
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline')
os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step4_mcfm.mcfm   import mcfm_v2
from step6_5_lora.lora_v2   import build_lora_blocks, freeze_trellis
from step6_5_lora.dual_path_v2 import dual_path_ctx_v2
from step8_decode_render.decode_render import (
    make_renderer, normalize_slat, decode_and_render,
    RENDER_RES, EXTRINSICS, INTRINSICS, SLAT_MEAN, SLAT_STD)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
ENH        = HERE.parent
RESULTS    = ENH / 'results_mcfm_v2_C_realgt_lora_seed6'
CKPT_PATH  = RESULTS / 'lora_ckpts' / 'lora_e018.pt'    # latest epoch
LOSS_JSON  = RESULTS / 'loss_history.json'
MCFM_DIR   = ENH / 'results_mcfm_v2_C_seed6_fixednoise' / 'beta0p0'
GT_DIR     = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                  '/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED = 'JeffreyXiang/TRELLIS-image-large'
DEVICE     = torch.device('cuda')
MODE       = 'v2_C'
N_FRAMES   = 150
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
LORA_RANK   = 4
RESCALE_T   = 3.0
_t_seq      = np.linspace(1, 0, STEPS + 1)
_t_seq      = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS     = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]
_DINO_NORM  = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

LOG_PATH = HERE / 'diag_lora_checkpoint.log'
LOG_PATH.unlink(missing_ok=True)   # fresh log each run

def log(msg=''):
    ts = f'[{time.strftime("%H:%M:%S")}] '
    full = ts + msg
    print(full, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(full + '\n')

def hline(title=''):
    log('─' * 72)
    if title:
        log(f'  {title}')
        log('─' * 72)


def get_window(frame_idx):
    # v2_C: current + next
    return [frame_idx, min(N_FRAMES, frame_idx + 1)]


def encode_frame(dino_model, frame_idx):
    img  = Image.open(GT_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)  # (1374, 1024)


def compute_khat_kpooled(raw_tokens, frame_idx):
    win  = get_window(frame_idx)
    lam  = torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)
    tok_d = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    K_hat, _ = mcfm_v2(tok_d, win, frame_idx, lam)
    stacked  = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0)
    weights  = torch.tensor([2.0 if i == frame_idx else 1.0 for i in win],
                             dtype=torch.float32, device=DEVICE)
    weights  /= weights.sum()
    K_pooled = (weights[:, None, None] * stacked).sum(0)
    return K_hat, K_pooled


def render_with_lora(flow_model, pipeline, lora_blocks, alpha_val,
                     raw_tokens, frame_idx, coords, fixed_noise_feats, renderer):
    """Render frame with LoRA at given alpha (scalar float)."""
    K_hat, K_pooled, = compute_khat_kpooled(raw_tokens, frame_idx)
    ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl  = K_hat.unsqueeze(0)
    alpha_p  = nn.Parameter(torch.tensor(float(alpha_val), device=DEVICE))
    flow_model.to(DEVICE)
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha_p, enhance_bias=None):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    slat = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    flow_model.to(DEVICE)
    return color.detach().clamp(0, 1)


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
        try:
            tw = draw.textbbox((0,0), lbl, font=font)[2]
        except Exception:
            tw = len(lbl) * 7
        draw.text((col*cell + (cell-tw)//2, 6), lbl, fill=(210,210,210), font=font)
    return canvas


def main():
    hline('LoRA Checkpoint Diagnostics')
    log(f'  checkpoint : {CKPT_PATH}')
    log(f'  mode       : {MODE}')
    log(f'  log        : {LOG_PATH}')

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    # ── DIAG 5 (no GPU): plot training loss curve ─────────────────────────────
    hline('DIAG 5 — Training loss curve (no GPU)')
    with open(LOSS_JSON) as f:
        history = json.load(f)
    epochs     = [r['epoch']            for r in history]
    task_loss  = [r['avg_task_loss']    for r in history]
    total_loss = [r['avg_total_loss']   for r in history]
    alphas     = [r['alpha']            for r in history]
    b_norms    = [r['lora_B_norm']      for r in history]
    ag_means   = [r.get('alpha_grad_mean', float('nan')) for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'LoRA v2_C realgt — 18 epochs', fontsize=12)
    axes[0].plot(epochs, task_loss,  'o-', color='#e06c75', lw=2, ms=5, label='task_loss')
    axes[0].plot(epochs, total_loss, 's--', color='#61afef', lw=1.5, ms=4, label='total_loss')
    axes[0].set_title('Task + Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, alphas, 'o-', color='#98c379', lw=2, ms=5)
    axes[1].axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='init=0.5')
    axes[1].set_title('alpha (should stay ≤ 0.5)'); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, b_norms, 'o-', color='#e5c07b', lw=2, ms=5)
    axes[2].set_title('LoRA B norm (growing = learning)'); axes[2].grid(True, alpha=0.3)
    valid_ag = [(e, g) for e, g in zip(epochs, ag_means) if not np.isnan(g)]
    if valid_ag:
        e_ag, g_ag = zip(*valid_ag)
        axes[3].bar(e_ag, g_ag, color=['#e06c75' if g > 0 else '#98c379' for g in g_ag], alpha=0.8)
        axes[3].axhline(0, color='black', lw=0.8)
    axes[3].set_title('alpha_grad_mean\n(+= reg pushing down ✓)'); axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    out5 = HERE / 'diag5_loss_curve.png'
    fig.savefig(out5, dpi=130, bbox_inches='tight')
    plt.close()
    log(f'  saved: {out5}')
    log(f'  epoch 1  alpha={alphas[0]:.4f}  task={task_loss[0]:.5f}  B_norm={b_norms[0]:.4f}')
    log(f'  epoch 18 alpha={alphas[-1]:.4f}  task={task_loss[-1]:.5f}  B_norm={b_norms[-1]:.4f}')
    if alphas[-1] > alphas[0]:
        log('  [WARNING] alpha GREW over training — unbounded growth confirmed')
    else:
        log(f'  [OK] alpha DECREASED from {alphas[0]:.4f} → {alphas[-1]:.4f}  (reg held it down)')

    # ── DIAG 2 (no GPU): print alpha from checkpoint ──────────────────────────
    hline('DIAG 2 — Alpha value from checkpoint')
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    alpha_ckpt = float(ckpt['alpha'].item())
    log(f'  epoch   : {ckpt["epoch"]}')
    log(f'  alpha   : {alpha_ckpt:.6f}')
    log(f'  avg_task_loss : {ckpt["avg_task_loss"]:.6f}')
    log(f'  gt_source     : {ckpt.get("gt_source", "unknown")}')
    if alpha_ckpt > 0.8:
        log('  [WARNING] alpha > 0.8 — highly likely unbounded growth is root cause')
    elif alpha_ckpt < 0.3:
        log(f'  [OK] alpha is small ({alpha_ckpt:.3f}) — alpha growth is NOT the primary issue')
    else:
        log(f'  [INFO] alpha={alpha_ckpt:.3f} — moderate, may still be too large')

    # ── DIAG 6 (no GPU): per-block LoRA weight norms ─────────────────────────
    hline('DIAG 6 — Per-block LoRA weight norms from checkpoint')
    lora_state = ckpt['lora_state']
    block_stats = {}
    for key, tensor in lora_state.items():
        # key like "0.lora_q.A", "0.lora_q.B", "0.lora_kv.A", etc.
        parts = key.split('.')
        blk_idx = int(parts[0])
        part_name = '.'.join(parts[1:])
        if blk_idx not in block_stats:
            block_stats[blk_idx] = {}
        block_stats[blk_idx][part_name] = {
            'norm': float(tensor.float().norm().item()),
            'max' : float(tensor.float().abs().max().item()),
            'shape': list(tensor.shape),
        }

    q_A_norms, q_B_norms, kv_A_norms, kv_B_norms = [], [], [], []
    for blk_idx in sorted(block_stats.keys()):
        stats = block_stats[blk_idx]
        qa  = stats.get('lora_q.A',  {}).get('norm', 0)
        qb  = stats.get('lora_q.B',  {}).get('norm', 0)
        kva = stats.get('lora_kv.A', {}).get('norm', 0)
        kvb = stats.get('lora_kv.B', {}).get('norm', 0)
        q_A_norms.append(qa); q_B_norms.append(qb)
        kv_A_norms.append(kva); kv_B_norms.append(kvb)
        log(f'  blk{blk_idx:02d}: lora_q.A={qa:.4f} lora_q.B={qb:.4f} '
            f'lora_kv.A={kva:.4f} lora_kv.B={kvb:.4f}')

    log(f'\n  Summary:')
    log(f'  lora_q.A  : min={min(q_A_norms):.4f}  max={max(q_A_norms):.4f}  mean={np.mean(q_A_norms):.4f}')
    log(f'  lora_q.B  : min={min(q_B_norms):.4f}  max={max(q_B_norms):.4f}  mean={np.mean(q_B_norms):.4f}')
    log(f'  lora_kv.A : min={min(kv_A_norms):.4f}  max={max(kv_A_norms):.4f}  mean={np.mean(kv_A_norms):.4f}')
    log(f'  lora_kv.B : min={min(kv_B_norms):.4f}  max={max(kv_B_norms):.4f}  mean={np.mean(kv_B_norms):.4f}')
    exploding = [i for i, n in enumerate(q_B_norms + kv_B_norms) if n > 10.0]
    if exploding:
        log(f'  [WARNING] B-matrix norms EXPLODING in blocks: {exploding}')
    elif max(q_B_norms + kv_B_norms) < 0.01:
        log('  [NOTE] B-matrix norms near zero — LoRA barely learned anything')
    else:
        log(f'  [OK] B-matrix norms in normal range')

    # Plot per-block norms
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('LoRA per-block weight norms (epoch 18)', fontsize=12)
    x = list(range(len(q_A_norms)))
    axes[0].bar(x, q_A_norms, alpha=0.7, label='lora_q.A',  color='#e06c75')
    axes[0].bar(x, q_B_norms, alpha=0.7, label='lora_q.B',  color='#98c379')
    axes[0].set_title('lora_q norms per block'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].bar(x, kv_A_norms, alpha=0.7, label='lora_kv.A', color='#61afef')
    axes[1].bar(x, kv_B_norms, alpha=0.7, label='lora_kv.B', color='#e5c07b')
    axes[1].set_title('lora_kv norms per block'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out6 = HERE / 'diag6_block_norms.png'
    fig.savefig(out6, dpi=130, bbox_inches='tight')
    plt.close()
    log(f'  saved: {out6}')

    # ── Heavy GPU diagnostics ─────────────────────────────────────────────────
    hline('Loading pipeline for GPU diagnostics (1, 3, 4)...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # Structure
    img_75 = Image.open(GT_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    log(f'  N_vox={N_vox}')
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    # Offload models not needed for flow
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # DINOv2 encode all 150 frames
    log(f'\n  Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino_model  = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens  = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            log(f'    {i}/{N_FRAMES}')
    dino_model.cpu(); torch.cuda.empty_cache()
    log('  DINOv2 done.')

    # Fixed noise
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)

    # Renderer
    renderer = make_renderer()

    # ── DIAG 3: K_pooled[1] vs K_pooled[150] vs raw_tokens ───────────────────
    hline('DIAG 3 — PATH B frame blindness: K_pooled[1] vs K_pooled[150]')
    K_hat_1,   Kp_1   = compute_khat_kpooled(raw_tokens, 1)
    K_hat_150, Kp_150 = compute_khat_kpooled(raw_tokens, 150)
    rt_1   = raw_tokens[1].to(DEVICE)
    rt_150 = raw_tokens[150].to(DEVICE)

    kp_diff   = (Kp_1.float() - Kp_150.float()).abs().mean().item()
    raw_diff  = (rt_1.float() - rt_150.float()).abs().mean().item()
    khat_diff = (K_hat_1.float() - K_hat_150.float()).abs().mean().item()

    log(f'  (K_pooled[1] - K_pooled[150]).abs().mean()  = {kp_diff:.6f}')
    log(f'  (raw_tokens[1] - raw_tokens[150]).abs().mean() = {raw_diff:.6f}')
    log(f'  (K_hat[1] - K_hat[150]).abs().mean()        = {khat_diff:.6f}')
    log(f'  ratio kp/raw  = {kp_diff/raw_diff:.4f}  (near 1 = varied, near 0 = blind)')
    log(f'  ratio khat/raw= {khat_diff/raw_diff:.4f}')
    if kp_diff / raw_diff < 0.1:
        log('  [CONFIRMED] PATH B IS BLIND — K_pooled[1] ≈ K_pooled[150] despite different frames')
    elif kp_diff / raw_diff > 0.8:
        log('  [OK] K_pooled varies enough between frames — frame blindness is NOT the issue')
    else:
        log('  [PARTIAL] K_pooled shows some frame variation but reduced')

    # Also check per-frame K_pooled variance
    kp_list = []
    for fi in [1, 25, 50, 75, 100, 125, 150]:
        _, kp = compute_khat_kpooled(raw_tokens, fi)
        kp_list.append(kp.float())
    kp_stack = torch.stack(kp_list, 0)   # (7, 1374, 1024)
    kp_var   = kp_stack.std(0).mean().item()
    rt_spot  = torch.stack([raw_tokens[fi].to(DEVICE).float() for fi in [1,25,50,75,100,125,150]])
    rt_var   = rt_spot.std(0).mean().item()
    log(f'\n  K_pooled std across 7 key frames : {kp_var:.6f}')
    log(f'  raw_tokens std across 7 key frames: {rt_var:.6f}')
    log(f'  K_pooled variation relative to raw: {kp_var/rt_var:.4f}')

    del K_hat_1, K_hat_150, Kp_1, Kp_150, rt_1, rt_150, kp_list, kp_stack, rt_spot
    gc.collect(); torch.cuda.empty_cache()

    # ── DIAG 4: K_hat vs raw_tokens across all frames ─────────────────────────
    hline('DIAG 4 — K_hat vs raw_tokens: is blending a no-op?')
    khat_raw_diffs = []
    for fi in range(1, N_FRAMES + 1):
        K_hat, _ = compute_khat_kpooled(raw_tokens, fi)
        rt = raw_tokens[fi].to(DEVICE)
        diff = (K_hat.float() - rt.float()).abs().mean().item()
        khat_raw_diffs.append(diff)
        del K_hat, rt
    gc.collect(); torch.cuda.empty_cache()

    mean_d = float(np.mean(khat_raw_diffs))
    max_d  = float(np.max(khat_raw_diffs))
    min_d  = float(np.min(khat_raw_diffs))
    log(f'  (K_hat[t] - raw_tokens[t]).abs().mean()  over all 150 frames:')
    log(f'    mean = {mean_d:.6f}')
    log(f'    max  = {max_d:.6f}')
    log(f'    min  = {min_d:.6f}')
    if mean_d < 1e-4:
        log('  [CONFIRMED] K_hat ≈ raw_tokens — blending IS a no-op (MCFM contributes nothing)')
    elif mean_d < 0.01:
        log(f'  [SMALL] K_hat differs from raw by {mean_d:.4f} — blending is very subtle')
    else:
        log(f'  [OK] K_hat differs from raw by {mean_d:.4f} — blending is active')

    # Plot K_hat vs raw diff per frame
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(1, N_FRAMES+1), khat_raw_diffs, 'o-', lw=1, ms=2, color='#e06c75', label='|K_hat-raw|.mean()')
    ax.axhline(mean_d, color='gray', ls='--', lw=1, label=f'mean={mean_d:.4f}')
    ax.set_xlabel('Frame'); ax.set_ylabel('Mean absolute diff')
    ax.set_title('DIAG 4: K_hat vs raw_tokens per frame\n(near-zero = blending is no-op)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out4 = HERE / 'diag4_khat_vs_raw.png'
    fig.savefig(out4, dpi=130, bbox_inches='tight')
    plt.close()
    log(f'  saved: {out4}')

    # ── DIAG 1: Force alpha=0, render f001/f075/f150 ─────────────────────────
    hline('DIAG 1 — Force alpha=0 on trained checkpoint → render f001/f075/f150')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=LORA_RANK, n_blocks=24).to(DEVICE)
    lora_blocks.load_state_dict(ckpt['lora_state'])
    flow_model.to(DEVICE)

    diag1_dir = HERE / 'diag1_alpha0_renders'
    diag1_dir.mkdir(exist_ok=True)

    for fi in [1, 75, 150]:
        log(f'\n  Rendering frame {fi} with alpha=0 (trained weights, LoRA contribution suppressed)...')
        t0 = time.time()
        color_a0 = render_with_lora(
            flow_model, pipeline, lora_blocks, alpha_val=0.0,
            raw_tokens=raw_tokens, frame_idx=fi,
            coords=coords, fixed_noise_feats=fixed_noise_feats, renderer=renderer)
        log(f'    alpha=0 done in {time.time()-t0:.1f}s')

        log(f'  Rendering frame {fi} with alpha={alpha_ckpt:.4f} (trained checkpoint)...')
        t0 = time.time()
        color_ckpt = render_with_lora(
            flow_model, pipeline, lora_blocks, alpha_val=alpha_ckpt,
            raw_tokens=raw_tokens, frame_idx=fi,
            coords=coords, fixed_noise_feats=fixed_noise_feats, renderer=renderer)
        log(f'    alpha_ckpt done in {time.time()-t0:.1f}s')

        # MCFM reference (pre-rendered)
        mcfm_path = MCFM_DIR / f'frame_{fi:04d}.png'
        gt_path   = GT_DIR   / f'frame_{fi:04d}.png'

        strip = make_strip([
            (gt_path,       'GT video'),
            (mcfm_path,     'MCFM ref (no LoRA)'),
            (color_a0,      f'LoRA alpha=0.0 (trained wts)'),
            (color_ckpt,    f'LoRA alpha={alpha_ckpt:.3f} (ckpt)'),
        ], font=font)
        out_path = diag1_dir / f'strip_f{fi:04d}.png'
        strip.save(out_path)
        log(f'  saved: {out_path}')

        # Pixel stats
        a0_arr    = color_a0.float().cpu()
        ckpt_arr  = color_ckpt.float().cpu()
        log(f'    alpha=0   : mean={a0_arr.mean():.4f}  std={a0_arr.std():.4f}  '
            f'  (near 1.0 = white teapot)')
        log(f'    alpha_ckpt: mean={ckpt_arr.mean():.4f}  std={ckpt_arr.std():.4f}')

        del color_a0, color_ckpt, a0_arr, ckpt_arr
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    hline('SUMMARY')
    log(f'  alpha at epoch 18   : {alpha_ckpt:.4f}  (started at 0.5)')
    log(f'  B-norm at epoch 18  : {b_norms[-1]:.4f}')
    log(f'  task_loss at e18    : {task_loss[-1]:.5f}  (started {task_loss[0]:.5f})')
    log(f'  K_pooled frame diff : kp_diff/raw_diff = {kp_diff/raw_diff:.4f}')
    log(f'  K_hat vs raw mean   : {mean_d:.6f}')
    log(f'\n  Check diag1_alpha0_renders/ for visual evidence of LoRA effect.')
    log(f'  If alpha=0 renders look good → LoRA weights are corrupting the output.')
    log(f'  If alpha=0 renders look same as MCFM ref → K_pooled path is the main damage.')
    log(f'\n  All outputs in: {HERE}')
    log('DONE.')


if __name__ == '__main__':
    main()
