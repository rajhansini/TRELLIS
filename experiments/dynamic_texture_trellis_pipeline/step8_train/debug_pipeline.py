"""
Debug Pipeline — dump every step's output for one target frame.

Run from step8_train/:
  SPCONV_ALGO=native ATTN_BACKEND=xformers python debug_pipeline.py [--frame 75]

Outputs under ../debug_results/:
  step1/              GT window frames + metadata
  step2/              DINO token PCA images (37×37 per frame) + cosine-sim stats
  step3/              lambda_vec bar chart + values
  step4/              K_hat PCA image + diff vs centre + stats
  step5/              token-assignment heatmap (37×37) + entropy histogram + stats
  step6/              E matrix stats
  step7/              SLaT feats trajectory plot over 25 steps  (mesh LoRA)
  step7_gaussian/     same, with gaussian LoRA (if checkpoint exists)
  step8_mesh/         rendered PNG + GT + side-by-side + pixel-diff heatmap
  step8_gaussian/     same, with gaussian decoder
"""

import os, sys, math, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']               = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']        = '1'
os.environ['TRANSFORMERS_OFFLINE']  = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.renderers import MeshRenderer, GaussianRenderer
import utils3d.torch as u3d

from step1_input_prep.input_prep       import N_FRAMES, load_frame, t_to_frame_idx, get_window_indices
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights  import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm                    import run_mcfm
from step5_3d_alignment.alignment       import extract_alignment
from step6_attn_enhancement.enhancement import build_enhancement_matrix
from step6_5_lora.lora                  import insert_lora
from step6_5_lora.ray_attention         import dual_path_ctx

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
RESULTS_DIR = Path('../results')
DEBUG_DIR   = Path('../debug_results')
CKPT_MESH   = RESULTS_DIR / 'lora_ckpts'
CKPT_GS     = RESULTS_DIR / 'lora_ckpts_gs'

K          = 3
LORA_RANK  = 4
NOISE_SEED = 42
RENDER_RES = 518
STEPS      = 25
RESCALE_T  = 3.0
DEVICE     = torch.device('cuda')

SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,  1.2039363384246826
], dtype=torch.float32)
SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338
], dtype=torch.float32)

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]

_fx_n           = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
MESH_INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                                 dtype=torch.float32)
MESH_EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32)

# ── Plot style ─────────────────────────────────────────────────────────────────
BG   = '#09101a'
SURF = '#111d2e'
BORD = '#1e3048'
TXT  = '#c8d8ea'
SUB  = '#5a7190'
C1   = '#22d3a0'  # works / green
C2   = '#3bb8f5'  # accent / blue
C3   = '#f4853a'  # risk / orange
C4   = '#a78bfa'  # purple

def _style_ax(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=SUB)
    for s in ax.spines.values():
        s.set_edgecolor(BORD)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)

# ── I/O helpers ───────────────────────────────────────────────────────────────

def out(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_txt(path, text):
    out(path).write_text(text)
    print(f'  [txt] {Path(path).name}')

def save_npy(path, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    np.save(out(path), arr)
    print(f'  [npy] {Path(path).name}  shape={arr.shape}')

def save_img(path, img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    img.save(out(path))
    print(f'  [img] {Path(path).name}')

def save_fig(path):
    plt.tight_layout()
    plt.savefig(out(path), dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [fig] {Path(path).name}')

def tstats(t, name=''):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float()
    else:
        t = torch.as_tensor(t).float()
    return (f'{name + " " if name else ""}shape={tuple(t.shape)}  '
            f'min={t.min().item():.5f}  max={t.max().item():.5f}  '
            f'mean={t.mean().item():.5f}  std={t.std().item():.5f}  '
            f'nan={t.isnan().sum().item()}  inf={t.isinf().sum().item()}')

# ── Viz helpers ────────────────────────────────────────────────────────────────

def _pca_to_rgb(mat_np):
    """(N, D) float32 numpy → (N, 3) float32 in [0,1] via PCA."""
    T = mat_np.astype(np.float32)
    T -= T.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(T, full_matrices=False)
    proj = T @ Vt[:3].T
    for i in range(3):
        lo, hi = proj[:, i].min(), proj[:, i].max()
        proj[:, i] = (proj[:, i] - lo) / (hi - lo + 1e-8)
    return proj.astype(np.float32)

def patch_pca_img(tokens_1374):
    """tokens_1374: (1374, 1024) tensor → uint8 (37, 37, 3) image."""
    patches = tokens_1374[5:].detach().cpu().float().numpy()   # (1369, 1024)
    rgb = _pca_to_rgb(patches).reshape(37, 37, 3)
    return (rgb * 255).clip(0, 255).astype(np.uint8)

def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)

# ── Latest checkpoint helper ───────────────────────────────────────────────────

def latest_ckpt(ckpt_dir):
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('lora_e*.pt'))
    return ckpts[-1] if ckpts else None

def load_lora_ckpt(flow_model, ckpt_path, label):
    lora_blocks, alpha = insert_lora(flow_model, rank=LORA_RANK)
    lora_blocks = lora_blocks.to(DEVICE)
    alpha       = nn.Parameter(torch.tensor(0.5, device=DEVICE))
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        lora_blocks.load_state_dict(ckpt['lora_state'])
        alpha.data.copy_(ckpt['alpha'].to(DEVICE))
        print(f'  [{label}] loaded {ckpt_path.name}  epoch={ckpt["epoch"]}  alpha={alpha.item():.4f}')
    else:
        print(f'  [{label}] no checkpoint found — using untrained LoRA (alpha=0.5)')
    lora_blocks.eval()
    return lora_blocks, alpha


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Input Prep
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(frame_num):
    d = DEBUG_DIR / 'step1'
    print(f'\n═══ STEP 1 — Input Prep ══════════════════════════════════')
    t_val     = (frame_num - 1) / (N_FRAMES - 1)
    frame_idx = t_to_frame_idx(t_val)
    win_idx   = get_window_indices(frame_idx, K)
    n_dupes   = len(win_idx) - len(set(win_idx))

    meta = [
        f'frame_num    = {frame_num}',
        f't_val        = {t_val:.6f}',
        f'frame_idx    = {frame_idx}',
        f'k            = {K}',
        f'window       = {win_idx}',
        f'duplicates   = {n_dupes}  (boundary clamping)',
    ]
    save_txt(d / 'metadata.txt', '\n'.join(meta))
    for line in meta:
        print(f'  {line}')

    for i, idx in enumerate(win_idx):
        offset = win_idx[i] - frame_idx
        label  = 'CENTER' if offset == 0 else f'offset{offset:+d}'
        save_img(d / f'f{idx:04d}_{label}.png', load_frame(idx))

    return t_val, frame_idx, win_idx


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DINO Encoding
# ══════════════════════════════════════════════════════════════════════════════

def run_step2(win_idx, frame_idx, dino):
    d = DEBUG_DIR / 'step2'
    print(f'\n═══ STEP 2 — DINO Encoding ═══════════════════════════════')

    # encode unique frames only
    unique_idx = list(dict.fromkeys(win_idx))
    raw = {}
    for idx in unique_idx:
        toks = encode_frames([idx], [load_frame(idx)], dino, DEVICE)
        raw[idx] = {k: v.cpu() for k, v in toks[idx].items()}

    stats_lines = []
    for idx in unique_idx:
        tokens = raw[idx]['tokens']  # (1374, 1024) on CPU
        label  = 'CENTER' if idx == frame_idx else f'f{idx:04d}'
        save_img(d / f'pca_{label}.png', patch_pca_img(tokens))
        stats_lines.append(tstats(tokens, f'tokens[{idx}]'))

    # Cosine similarity of each frame to the centre frame
    ctr = F.normalize(raw[frame_idx]['tokens'].float(), dim=-1)  # (1374, 1024)
    sim_lines = ['', '─ cosine similarity to centre (mean over 1374 tokens) ─']
    for idx in unique_idx:
        t   = F.normalize(raw[idx]['tokens'].float(), dim=-1)
        sim = (t * ctr).sum(-1).mean().item()
        marker = ' ← centre' if idx == frame_idx else ''
        sim_lines.append(f'  frame {idx:4d}: {sim:.6f}{marker}')

    save_txt(d / 'stats.txt', '\n'.join(stats_lines + sim_lines))

    # Rebuild full dict including boundary duplicates
    tokens_gpu = {idx: {k: v.to(DEVICE) for k, v in raw[idx].items()}
                  for idx in win_idx}  # duplicates map to same entry
    return tokens_gpu


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Frame Weights
# ══════════════════════════════════════════════════════════════════════════════

def run_step3(t_val, frame_idx, win_idx, tokens_gpu):
    d = DEBUG_DIR / 'step3'
    print(f'\n═══ STEP 3 — Frame Weights ════════════════════════════════')

    projector = FrameWeightProjector(k=K).to(DEVICE)
    lambda_vec, lambda_dict = get_frame_weights(t_val, frame_idx, win_idx, tokens_gpu, projector)

    lines = [
        f'lambda_vec  = {[round(v, 6) for v in lambda_vec.tolist()]}',
        f'sum         = {lambda_vec.sum().item():.6f}',
        f'lambda[k={K}] = {lambda_vec[K].item():.6f}  ← center frame weight',
        '', '─ per-frame weights (after boundary accumulation) ─',
    ]
    for fi, lw in lambda_dict.items():
        marker = ' ← centre' if fi == frame_idx else ''
        lines.append(f'  frame {fi:4d}: {lw.item():.6f}{marker}')
    save_txt(d / 'lambda.txt', '\n'.join(lines))

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(BG)
    xs     = list(range(len(win_idx)))
    colors = [C1 if i == K else C2 for i in xs]
    ax.bar(xs, lambda_vec.cpu().tolist(), color=colors, edgecolor=BORD, linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in win_idx], fontsize=8)
    ax.set_xlabel('frame index')
    ax.set_ylabel('λ weight')
    ax.set_title(f'Frame weights — uniform λ = 1/{len(win_idx)} = {lambda_vec[0].item():.4f}')
    ax.axvline(K - 0.5 + 0.5, color=C3, linewidth=1.2, linestyle='--', alpha=0.7,
               label=f'centre idx={frame_idx}')
    ax.legend(fontsize=8, facecolor=SURF, labelcolor=TXT)
    _style_ax(ax)
    save_fig(d / 'lambda_bar.png')

    return lambda_vec, projector


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MCFM
# ══════════════════════════════════════════════════════════════════════════════

def run_step4(tokens_gpu, win_idx, frame_idx, lambda_vec):
    d = DEBUG_DIR / 'step4'
    print(f'\n═══ STEP 4 — MCFM v2 ════════════════════════════════════')

    K_hat, _ = run_mcfm('v2', tokens_gpu, win_idx, frame_idx, lambda_vec)  # (1374, 1024)
    centre   = tokens_gpu[frame_idx]['tokens'].float().cpu()                # (1374, 1024)

    save_img(d / 'K_hat_pca.png', patch_pca_img(K_hat))
    save_img(d / 'centre_pca.png', patch_pca_img(centre))

    diff = (K_hat.float().cpu() - centre).abs()
    save_img(d / 'diff_abs_pca.png', patch_pca_img(diff))
    save_npy(d / 'K_hat.npy', K_hat)

    # Per-token cosine sim: K_hat vs centre
    k_norm = F.normalize(K_hat.float().cpu(), dim=-1)
    c_norm = F.normalize(centre, dim=-1)
    cos_sim = (k_norm * c_norm).sum(-1)  # (1374,)

    lines = [
        tstats(K_hat, 'K_hat (output of MCFM)'),
        tstats(centre, 'centre frame tokens (raw)'),
        tstats(diff, 'abs(K_hat - centre)'),
        '',
        f'cosine_sim(K_hat, centre) per token:',
        f'  mean = {cos_sim.mean().item():.5f}',
        f'  min  = {cos_sim.min().item():.5f}',
        f'  max  = {cos_sim.max().item():.5f}',
        f'  std  = {cos_sim.std().item():.5f}',
        f'',
        f'How much MCFM changed the tokens relative to centre-frame raw:',
        f'  diff mean/dim = {diff.mean().item():.5f}',
        f'  If cos_sim ≈ 1.0 everywhere → MCFM barely changed the tokens (expected for uniform λ)',
    ]
    save_txt(d / 'stats.txt', '\n'.join(lines))
    print(f'  K_hat cosine_sim to centre: mean={cos_sim.mean().item():.4f}')

    # Cosine sim histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(BG)
    ax.hist(cos_sim.numpy(), bins=60, color=C2, edgecolor=BG, linewidth=0.3)
    ax.axvline(cos_sim.mean().item(), color=C3, linestyle='--', linewidth=1.2,
               label=f'mean={cos_sim.mean().item():.4f}')
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('# tokens')
    ax.set_title('K_hat vs centre frame tokens — per-token cosine similarity')
    ax.legend(fontsize=8, facecolor=SURF, labelcolor=TXT)
    _style_ax(ax)
    save_fig(d / 'cosine_sim_hist.png')

    return K_hat


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 3D Alignment
# ══════════════════════════════════════════════════════════════════════════════

def run_step5(flow_model, coords, K_hat_gpu, N_vox):
    d = DEBUG_DIR / 'step5'
    print(f'\n═══ STEP 5 — 3D Alignment ════════════════════════════════')

    cond = K_hat_gpu.unsqueeze(0)  # (1, 1374, 1024)
    torch.manual_seed(NOISE_SEED)
    sx = sp.SparseTensor(
        feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
        coords=coords,
    )
    with torch.no_grad():
        attn_map, _, voxel_to_token = extract_alignment(flow_model, sx, cond, DEVICE)

    N_vox_ca = voxel_to_token.shape[0]
    print(f'  N_vox_input={N_vox}  N_vox_ca={N_vox_ca}  '
          f'(downsampled inside G_L cross-attn blocks)')

    save_npy(d / 'voxel_to_token.npy', voxel_to_token)

    v2t     = voxel_to_token.numpy()
    attn_np = attn_map.float().numpy()  # (N_vox_ca, 1374)

    # ── Attention mass breakdown ───────────────────────────────────────────────
    # How much probability mass goes to CLS / REG / patches on average?
    # This tells us whether patch tokens have meaningful signal after the argmax fix.
    mass_cls     = attn_np[:, 0].mean()
    mass_reg     = attn_np[:, 1:5].sum(axis=1).mean()
    mass_patches = attn_np[:, 5:].sum(axis=1).mean()
    max_patch_per_voxel = attn_np[:, 5:].max(axis=1)   # (N_vox_ca,) best patch attn per voxel
    max_reg_per_voxel   = attn_np[:, 1:5].max(axis=1)  # (N_vox_ca,) best REG attn per voxel
    print(f'  mass → CLS={mass_cls:.4f}  REG={mass_reg:.4f}  patches={mass_patches:.4f}')
    print(f'  max patch attn per voxel: mean={max_patch_per_voxel.mean():.5f}  '
          f'max={max_patch_per_voxel.max():.5f}')
    print(f'  max REG   attn per voxel: mean={max_reg_per_voxel.mean():.5f}  '
          f'max={max_reg_per_voxel.max():.5f}')
    print(f'  ratio max_patch/max_REG:  {(max_patch_per_voxel.mean()/max(max_reg_per_voxel.mean(),1e-9)):.4f}')

    # ── Attention entropy ──────────────────────────────────────────────────────
    eps     = 1e-9
    entropy = -(attn_np * np.log(attn_np + eps)).sum(axis=1)  # (N_vox_ca,)
    max_ent = math.log(1374)
    focused = 1.0 - entropy.mean() / max_ent   # 0 = uniform, 1 = one-hot

    fig_e, ax_e = plt.subplots(figsize=(7, 3))
    fig_e.patch.set_facecolor(BG)
    ax_e.hist(entropy, bins=80, color=C1, edgecolor=BG, linewidth=0.2)
    ax_e.axvline(max_ent,         color=C3, linestyle='--', linewidth=1.2,
                 label=f'max entropy = {max_ent:.2f}  (uniform)')
    ax_e.axvline(entropy.mean(),  color=C2, linestyle='--', linewidth=1.2,
                 label=f'mean = {entropy.mean():.2f}  (focusedness={focused:.3f})')
    ax_e.set_xlabel('attention entropy'); ax_e.set_ylabel('voxel count')
    ax_e.set_title('Per-voxel attention entropy  (lower → more focused on one token)')
    ax_e.legend(fontsize=8, facecolor=SURF, labelcolor=TXT)
    _style_ax(ax_e)
    save_fig(d / 'entropy_histogram.png')

    # ── Token assignment heatmap (patch tokens → 37×37) ───────────────────────
    patch_counts = np.zeros(1369, dtype=np.float32)
    for ti_flat in range(1369):
        patch_counts[ti_flat] = (v2t == (ti_flat + 5)).sum()
    heatmap = patch_counts.reshape(37, 37)

    fig_h, ax_h = plt.subplots(figsize=(6, 6))
    fig_h.patch.set_facecolor(BG)
    im = ax_h.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax_h.set_title(f'Voxels assigned per patch token (argmax)\n'
                   f'37×37 = DINOv2 patch grid at 518×518')
    plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    ax_h.set_facecolor(BG)
    ax_h.title.set_color(TXT)
    ax_h.tick_params(colors=SUB)
    save_fig(d / 'token_assignment_heatmap.png')

    # ── Stats text ─────────────────────────────────────────────────────────────
    cls_count   = int((v2t == 0).sum())
    reg_count   = int(sum((v2t == i).sum() for i in range(1, 5)))
    patch_count = int((v2t >= 5).sum())
    top10       = np.argsort(patch_counts)[::-1][:10]

    lines = [
        f'N_vox_input = {N_vox}',
        f'N_vox_ca    = {N_vox_ca}  (cross-attn level, after G_L downsampling)',
        f'N_ctx       = 1374',
        f'',
        f'Attention mass (mean probability per voxel):',
        f'  → CLS  (1 token)    : {mass_cls:.5f}  total',
        f'  → REG  (4 tokens)   : {mass_reg:.5f}  total  ({mass_reg/4:.5f} avg per REG token)',
        f'  → patches (1369)    : {mass_patches:.5f}  total  ({mass_patches/1369:.7f} avg per patch)',
        f'',
        f'Max attn to best patch vs best REG per voxel:',
        f'  max patch attn  mean={max_patch_per_voxel.mean():.5f}  max={max_patch_per_voxel.max():.5f}',
        f'  max REG   attn  mean={max_reg_per_voxel.mean():.5f}  max={max_reg_per_voxel.max():.5f}',
        f'  ratio (patch/REG) = {max_patch_per_voxel.mean()/max(max_reg_per_voxel.mean(),1e-9):.4f}',
        f'  → ratio > 0.1 means patch signal is meaningful',
        f'  → ratio < 0.01 means patch assignments are noise',
        f'',
        f'Assignment breakdown (argmax over patch tokens only, post-fix):',
        f'  → CLS token (0)     : {cls_count} voxels',
        f'  → REG tokens (1-4)  : {reg_count} voxels',
        f'  → patch tokens (5+) : {patch_count} voxels',
        f'',
        f'Attention entropy:',
        f'  mean       = {entropy.mean():.4f}',
        f'  std        = {entropy.std():.4f}',
        f'  min/max    = {entropy.min():.4f} / {entropy.max():.4f}',
        f'  max_possible = {max_ent:.4f}  (perfectly uniform = log(1374))',
        f'  focusedness  = {focused:.4f}  (0=uniform, 1=one-hot)',
        f'',
        f'Top 10 most-assigned patch tokens:',
    ]
    for ti_flat in top10:
        tok_idx      = ti_flat + 5
        row, col     = divmod(int(ti_flat), 37)
        count        = int(patch_counts[ti_flat])
        pct          = count / N_vox_ca * 100
        lines.append(f'  token {tok_idx:5d}  patch[{row:2d},{col:2d}]  → {count:6d} voxels  ({pct:.1f}%)')

    save_txt(d / 'stats.txt', '\n'.join(lines))
    print(f'  entropy mean={entropy.mean():.3f}  focusedness={focused:.3f}')
    print(f'  CLS={cls_count}  REG={reg_count}  patches={patch_count}')

    return voxel_to_token


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Enhancement Matrix E
# ══════════════════════════════════════════════════════════════════════════════

def run_step6(voxel_to_token):
    d = DEBUG_DIR / 'step6'
    print(f'\n═══ STEP 6 — Enhancement Matrix E ════════════════════════')

    E = build_enhancement_matrix(voxel_to_token, N_ctx=1374)

    N_vox_ca, N_ctx = E.shape
    n_finite = (E == 0.0).sum().item()  # assigned entries (E=0 = keep logit)
    n_masked = torch.isinf(E).sum().item()  # -inf entries (zeroed in softmax)
    sparsity = n_masked / (N_vox_ca * N_ctx)

    lines = [
        f'E shape          = {tuple(E.shape)}  (N_vox_ca × N_ctx)',
        f'E formulation    = additive log-space mask (A_raw + E)',
        f'assigned entries = {n_finite}  E[v,assigned]=0.0  (logit unchanged)',
        f'masked  entries  = {n_masked}  E[v,others]=-inf   (exp(-inf)=0, gone in softmax)',
        f'mask fraction    = {sparsity:.8f}',
        f'',
        f'What happens at attention time:',
        f'  softmax(A_raw + E) with E[v,assigned]=0, E[v,others]=-inf',
        f'  → non-assigned tokens collapse to 0 in softmax',
        f'  → 100% attention mass routed to assigned token per voxel',
        f'',
        f'Implication for training:',
        f'  If assigned token is correct  → focused, useful signal',
        f'  If assigned token is wrong    → wrong semantic all training long',
        f'  The assignment is FIXED (computed once at frame 75)',
    ]
    save_txt(d / 'stats.txt', '\n'.join(lines))
    for line in lines:
        print(f'  {line}')

    return E


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Denoising Trajectory
# ══════════════════════════════════════════════════════════════════════════════

def run_step7(flow_model, coords, cond_gl, K_pooled, lora_blocks, alpha, E, N_vox,
              suffix='mesh'):
    d = DEBUG_DIR / f'step7_{suffix}'
    print(f'\n═══ STEP 7 — Denoising ({suffix}) ════════════════════════')

    torch.manual_seed(NOISE_SEED)
    x = sp.SparseTensor(
        feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
        coords=coords,
    )

    norms, abs_means, fmins, fmaxs = [], [], [], []

    def _record(feats):
        f = feats.detach().cpu().float()
        norms.append(f.norm(dim=-1).mean().item())
        abs_means.append(f.abs().mean().item())
        fmins.append(f.min().item())
        fmaxs.append(f.max().item())

    _record(x.feats)  # t=1.0 initial noise

    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
            _record(x.feats)

    step_xs = list(range(STEPS + 1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f'SLaT feats over {STEPS} denoising steps  ({suffix} LoRA)',
                 color=TXT, fontsize=11)
    fig.patch.set_facecolor(BG)

    panels = [
        (axes[0, 0], norms,     'per-voxel norm (mean)',    C1),
        (axes[0, 1], abs_means, '|feats| mean',             C2),
        (axes[1, 0], fmaxs,     'feats max',                C3),
        (axes[1, 1], fmins,     'feats min',                C4),
    ]
    for ax, vals, title, color in panels:
        ax.plot(step_xs, vals, color=color, linewidth=1.6)
        ax.set_xlabel('step')
        ax.set_title(title, fontsize=9)
        _style_ax(ax)

    save_fig(d / 'trajectory.png')

    # Per-step stats as text (every 5 steps)
    lines = [f'{"step":>5}  {"t":>8}  {"norm":>10}  {"abs_mean":>10}  {"max":>10}  {"min":>10}']
    for i in range(0, STEPS + 1, 5):
        t_here = T_PAIRS[i][0] if i < STEPS else 0.0
        lines.append(f'{i:>5}  {t_here:>8.4f}  '
                     f'{norms[i]:>10.5f}  {abs_means[i]:>10.5f}  '
                     f'{fmaxs[i]:>10.5f}  {fmins[i]:>10.5f}')
    save_txt(d / 'trajectory_stats.txt', '\n'.join(lines))

    print(f'  start: norm={norms[0]:.4f}  end: norm={norms[-1]:.4f}  '
          f'max={fmaxs[-1]:.4f}  min={fmins[-1]:.4f}')

    return x  # unnormalized SLaT at t=0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Mesh
# ══════════════════════════════════════════════════════════════════════════════

def run_step8_mesh(pipeline, slat_x0, frame_num):
    d = DEBUG_DIR / 'step8_mesh'
    print(f'\n═══ STEP 8 — Mesh Decode + Render ════════════════════════')

    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1},
    )
    extr = MESH_EXTRINSICS.to(DEVICE)
    intr = MESH_INTRINSICS.to(DEVICE)

    slat = normalize_slat(slat_x0)

    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]

        if hasattr(mesh, 'vertices') and mesh.vertices is not None:
            save_txt(d / 'mesh_stats.txt', '\n'.join([
                tstats(mesh.vertices,     'vertices'),
                tstats(mesh.vertex_attrs, 'vertex_attrs'),
                f'N_vertices = {mesh.vertices.shape[0]}',
                f'N_faces    = {mesh.faces.shape[0] if hasattr(mesh, "faces") else "N/A"}',
            ]))

        result  = renderer.render(mesh, extr, intr, return_types=['color', 'mask'])

    mask     = result['mask'].unsqueeze(0)
    color    = result['color'] * mask + (1.0 - mask)
    color_np = (color.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    mask_np  = (mask.detach().cpu().squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)

    save_img(d / 'rendered.png', color_np)
    save_img(d / 'mask.png', mask_np)

    gt     = np.array(load_frame(frame_num).resize((RENDER_RES, RENDER_RES)))
    save_img(d / 'gt.png', gt)
    save_img(d / 'comparison.png', np.concatenate([gt, color_np], axis=1))

    diff = np.abs(gt.astype(np.float32) - color_np.astype(np.float32)).mean(axis=2)
    fig_d, ax_d = plt.subplots(figsize=(5, 5))
    fig_d.patch.set_facecolor(BG)
    im = ax_d.imshow(diff, cmap='hot')
    ax_d.set_title(f'pixel diff  mean={diff.mean():.2f}', color=TXT)
    plt.colorbar(im, ax=ax_d)
    ax_d.tick_params(colors=SUB)
    save_fig(d / 'pixel_diff.png')

    gt_f   = gt.astype(np.float32) / 255
    rend_f = color_np.astype(np.float32) / 255
    mse    = ((gt_f - rend_f) ** 2).mean()
    psnr   = -10 * math.log10(mse + 1e-10)
    save_txt(d / 'metrics.txt',
             f'MSE  = {mse:.6f}\nPSNR = {psnr:.2f} dB')
    print(f'  MSE={mse:.6f}  PSNR={psnr:.2f} dB')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Gaussian
# ══════════════════════════════════════════════════════════════════════════════

def run_step8_gaussian(pipeline, slat_x0, frame_num):
    d = DEBUG_DIR / 'step8_gaussian'
    print(f'\n═══ STEP 8 — Gaussian Decode + Render ════════════════════')

    if 'slat_decoder_gs' not in pipeline.models:
        print('  slat_decoder_gs not in pipeline.models — skipping gaussian step')
        return

    gs_decoder = pipeline.models['slat_decoder_gs']
    gs_decoder.to(DEVICE)
    gs_decoder.convert_to_fp32()
    gs_decoder.dtype = torch.float32

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = RENDER_RES
    renderer.rendering_options.bg_color   = (1.0, 1.0, 1.0)
    renderer.rendering_options.near       = 0.8
    renderer.rendering_options.far        = 1.6

    fov  = torch.deg2rad(torch.tensor(40.)).to(DEVICE)
    eye  = torch.tensor([0., 0., 2.]).to(DEVICE)
    tgt  = torch.zeros(3).to(DEVICE)
    up   = torch.tensor([0., 1., 0.]).to(DEVICE)
    extr = u3d.extrinsics_look_at(eye, tgt, up)
    intr = u3d.intrinsics_from_fov_xy(fov, fov)

    slat = normalize_slat(slat_x0)

    with torch.no_grad():
        decoded  = pipeline.decode_slat(slat, ['gaussian'])
        gaussian = decoded['gaussian'][0]

        # Sanitize (same logic as train_gaussian.py)
        for attr in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc']:
            t = getattr(gaussian, attr, None)
            if t is not None:
                setattr(gaussian, attr, torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))
        gaussian._scaling = gaussian._scaling.clamp(-5.0, 3.0)

        if hasattr(gaussian, 'rots_bias'):
            rot_eff   = gaussian._rotation + gaussian.rots_bias[None, :]
            zero_mask = (rot_eff.norm(dim=-1, keepdim=True) < 1e-6).expand_as(gaussian._rotation)
            gaussian._rotation = torch.where(zero_mask, torch.zeros_like(gaussian._rotation),
                                             gaussian._rotation)

        cov3d = gaussian.get_covariance()
        cov3d = torch.nan_to_num(cov3d, nan=0.0, posinf=1e-4, neginf=0.0).clamp(-1.0, 1.0)
        gaussian.get_covariance = lambda *a, **kw: cov3d

        # Save gaussian stats before rendering
        try:
            gs_stats = [
                tstats(gaussian._xyz,         '_xyz'),
                tstats(gaussian._scaling,      '_scaling'),
                tstats(gaussian._rotation,     '_rotation'),
                tstats(gaussian._opacity,      '_opacity'),
                tstats(gaussian._features_dc,  '_features_dc'),
                tstats(cov3d,                  'cov3d'),
                f'N_gaussians = {gaussian._xyz.shape[0]}',
            ]
            save_txt(d / 'gaussian_stats.txt', '\n'.join(gs_stats))
        except Exception as e:
            print(f'  gaussian stats failed: {e}')

        result   = renderer.render(gaussian, extr, intr)
        color    = result['color']

    color_np = (color.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    save_img(d / 'rendered.png', color_np)

    gt     = np.array(load_frame(frame_num).resize((RENDER_RES, RENDER_RES)))
    save_img(d / 'gt.png', gt)
    save_img(d / 'comparison.png', np.concatenate([gt, color_np], axis=1))

    diff = np.abs(gt.astype(np.float32) - color_np.astype(np.float32)).mean(axis=2)
    fig_d, ax_d = plt.subplots(figsize=(5, 5))
    fig_d.patch.set_facecolor(BG)
    im = ax_d.imshow(diff, cmap='hot')
    ax_d.set_title(f'pixel diff  mean={diff.mean():.2f}', color=TXT)
    plt.colorbar(im, ax=ax_d)
    ax_d.tick_params(colors=SUB)
    save_fig(d / 'pixel_diff.png')

    gt_f   = gt.astype(np.float32) / 255
    rend_f = color_np.astype(np.float32) / 255
    mse    = ((gt_f - rend_f) ** 2).mean()
    psnr   = -10 * math.log10(mse + 1e-10)
    save_txt(d / 'metrics.txt',
             f'MSE  = {mse:.6f}\nPSNR = {psnr:.2f} dB')
    print(f'  MSE={mse:.6f}  PSNR={psnr:.2f} dB')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Debug: dump every step output for one frame.')
    parser.add_argument('--frame', type=int, default=75,
                        help='Target frame 1-150 (default 75)')
    args = parser.parse_args()

    frame_num = args.frame
    print(f'=== Debug Pipeline  frame={frame_num}  output={DEBUG_DIR.resolve()} ===')

    # ── Steps 1-4: DINO only ──────────────────────────────────────────────────
    t_val, frame_idx, win_idx = run_step1(frame_num)

    print('\nLoading DINOv2...')
    dino       = load_dino(DEVICE)
    tokens_gpu = run_step2(win_idx, frame_idx, dino)
    del dino; gc.collect(); torch.cuda.empty_cache()

    lambda_vec, _ = run_step3(t_val, frame_idx, win_idx, tokens_gpu)
    K_hat         = run_step4(tokens_gpu, win_idx, frame_idx, lambda_vec)

    K_pooled = torch.stack([tokens_gpu[i]['tokens']
                            for i in win_idx]).mean(0)  # (1374, 1024)

    # ── Load pipeline ──────────────────────────────────────────────────────────
    print('\nLoading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    print('Sampling voxel structure from frame 75...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    del cond_struct; gc.collect(); torch.cuda.empty_cache()
    print(f'  N_vox = {N_vox}')

    # Offload everything except flow_model to free VRAM for steps 5-7
    for name, model in list(pipeline.models.items()):
        if name != 'slat_flow_model':
            try:
                model.cpu()
            except Exception:
                pass
    torch.cuda.empty_cache()

    # ── Steps 5-6 ─────────────────────────────────────────────────────────────
    voxel_to_token = run_step5(flow_model, coords, K_hat.to(DEVICE), N_vox)
    E              = run_step6(voxel_to_token)

    # ── Load LoRA checkpoints ─────────────────────────────────────────────────
    flow_model.eval()
    cond_gl = K_hat.unsqueeze(0).to(DEVICE)  # (1, 1374, 1024)

    ckpt_mesh_path = latest_ckpt(CKPT_MESH)
    ckpt_gs_path   = latest_ckpt(CKPT_GS)

    print(f'\nMesh LoRA:     {ckpt_mesh_path or "NONE"}')
    print(f'Gaussian LoRA: {ckpt_gs_path   or "NONE"}')

    # ── Step 7 mesh trajectory ────────────────────────────────────────────────
    lora_mesh, alpha_mesh = load_lora_ckpt(flow_model, ckpt_mesh_path, 'mesh')
    slat_mesh = run_step7(flow_model, coords, cond_gl, K_pooled,
                          lora_mesh, alpha_mesh, E, N_vox, suffix='mesh')

    # ── Step 7 gaussian trajectory (separate LoRA if available) ──────────────
    if ckpt_gs_path is not None:
        lora_gs, alpha_gs = load_lora_ckpt(flow_model, ckpt_gs_path, 'gaussian')
        slat_gs = run_step7(flow_model, coords, cond_gl, K_pooled,
                            lora_gs, alpha_gs, E, N_vox, suffix='gaussian')
    else:
        print('\n  No gaussian LoRA — reusing mesh SLaT for gaussian decode')
        slat_gs = slat_mesh

    # ── Step 8 mesh ───────────────────────────────────────────────────────────
    if 'slat_decoder_mesh' in pipeline.models:
        pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    run_step8_mesh(pipeline, slat_mesh, frame_num)

    flow_model.to(DEVICE)
    if 'slat_decoder_mesh' in pipeline.models:
        pipeline.models['slat_decoder_mesh'].cpu()
    gc.collect(); torch.cuda.empty_cache()

    # ── Step 8 gaussian ───────────────────────────────────────────────────────
    flow_model.cpu(); gc.collect(); torch.cuda.empty_cache()

    try:
        run_step8_gaussian(pipeline, slat_gs, frame_num)
    except Exception as e:
        print(f'  step8_gaussian failed: {e}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n=== Done ===')
    print(f'All outputs in: {DEBUG_DIR.resolve()}')
    print('')
    for p in sorted(DEBUG_DIR.rglob('*')):
        if p.is_file():
            size = p.stat().st_size
            size_str = f'{size/1024:.0f}K' if size < 1e6 else f'{size/1e6:.1f}M'
            print(f'  {str(p.relative_to(DEBUG_DIR)):60s}  {size_str}')


if __name__ == '__main__':
    main()
