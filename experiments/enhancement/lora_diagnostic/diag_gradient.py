"""
Gradient diagnostics for LoRA training — 5 plots, run in order 5→1→2→3→4.

Diag 5: ||B|| trajectory under eps=1e-16, wd=0 (2 epochs, 3 frames)
Diag 1: Per-block gradient norm (log scale) — where does signal die?
Diag 2: Per-denoising-step gradient norm (all 25 steps, backprop through all)
Diag 3: B.grad per block for frames 1, 75, 150 — frame variation
Diag 4: Adam effective step size at eps=1e-8 vs eps=1e-16 (analytical)

Output: lora_diagnostic/diag{1..5}_*.png + diag_gradient.log
"""

import sys, os
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

_HERE = Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent.parent
_PIPE = _EXP.parent / 'dynamic_texture_trellis_pipeline'

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

_LOG = open(_HERE / 'diag_gradient.log', 'w', buffering=1)

class _Tee:
    def write(self, m): sys.__stdout__.write(m); _LOG.write(m)
    def flush(self): sys.__stdout__.flush(); _LOG.flush()

sys.stdout = _Tee()
sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step4_mcfm.mcfm           import mcfm_v2
from step6_5_lora.lora_v2_fixed import build_lora_blocks, freeze_trellis, trainable_params
from step6_5_lora.dual_path_v2  import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render, RENDER_RES)

# ── constants ─────────────────────────────────────────────────────────────────
VIDEO_DIR  = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                  '/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED = 'JeffreyXiang/TRELLIS-image-large'
DEVICE     = torch.device('cuda')
N_FRAMES   = 150
STRUCT_SEED = 42
FIXED_SEED  = 6
STEPS       = 25
RESCALE_T   = 3.0
LORA_RANK   = 4
ALPHA       = 0.1
LOSS_SCALE  = 4096.0
MODE        = 'v2_C'

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── helpers ───────────────────────────────────────────────────────────────────
def encode_frame(dino_model, idx):
    img = Image.open(VIDEO_DIR / f'frame_{idx:04d}.png').convert('RGB').resize((518,518), Image.LANCZOS)
    x   = _DINO_NORM(torch.from_numpy(np.array(img).astype(np.float32)/255).permute(2,0,1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        toks = F.layer_norm((t := dino_model(x, is_training=True)['x_prenorm']), t.shape[-1:])
    return toks.squeeze(0)

def load_gt(idx):
    img = Image.open(VIDEO_DIR / f'frame_{idx:04d}.png').convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float().div(255).permute(2,0,1).to(DEVICE)

def blend_v2c(raw_tokens, idx):
    win = [idx, min(N_FRAMES, idx+1)]
    lam = torch.ones(len(win), dtype=torch.float32, device=DEVICE) / len(win)
    tok_d = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    K_hat, _ = mcfm_v2(tok_d, win, idx, lam)
    stacked   = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0)
    w = torch.tensor([2.0 if i == idx else 1.0 for i in win], dtype=torch.float32, device=DEVICE)
    w /= w.sum()
    K_pool = (w[:,None,None] * stacked).sum(0)
    return K_hat, K_pool

def denoise_prefix_nograd(flow_model, x, cond_gl, K_pool, lora_blocks):
    with torch.no_grad():
        for t, t_prev in T_PAIRS[:-1]:
            t_ten = torch.tensor([1000.*t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pool, lora_blocks, ALPHA):
                v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x

def denoise_last_with_grad(flow_model, x_in, cond_gl, K_pool, lora_blocks):
    t, t_prev = T_PAIRS[-1]
    t_ten = torch.tensor([1000.*t], device=DEVICE, dtype=torch.float32)
    with dual_path_ctx_v2(flow_model, K_pool, lora_blocks, ALPHA):
        v = flow_model(x_in, t_ten, cond_gl)
    return x_in.replace(x_in.feats - (t - t_prev) * v.feats)

def lora_b_norms(lora_blocks):
    return [blk.lora_q.B.float().norm().item() for blk in lora_blocks]

def lora_b_grads(lora_blocks):
    norms = []
    for blk in lora_blocks:
        g = blk.lora_q.B.grad
        norms.append(g.float().norm().item() if g is not None else 0.0)
    return norms

# ── setup ─────────────────────────────────────────────────────────────────────
print('\n[SETUP] Loading pipeline...')
pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
pipeline.to(DEVICE)   # all models on GPU for struct sampling

print('[SETUP] Struct sampling (pipeline.get_cond → sample_sparse_structure)...')
img_75      = Image.open(VIDEO_DIR / 'frame_0075.png').convert('RGB')
cond_struct = pipeline.get_cond([img_75])
torch.manual_seed(STRUCT_SEED)
with torch.no_grad():
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
N_vox = coords.shape[0]
print(f'  N_vox={N_vox}')
assert N_vox == 7301, f'N_vox={N_vox} != 7301'
del cond_struct, img_75; gc.collect(); torch.cuda.empty_cache()

# keep only the three models we actually need on GPU
flow_model = pipeline.models['slat_flow_model']
for n, m in pipeline.models.items():
    if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
        try: m.cpu()
        except: pass
torch.cuda.empty_cache()

print('[SETUP] Encoding DINOv2 frames (1, 2, 75, 76, 150)...')
dino = pipeline.models['image_cond_model']   # already on DEVICE
raw_tokens = {}
# blend_v2c window is [idx, idx+1]; encode neighbors so tok_d lookup never misses
for idx in [1, 2, 75, 76, 150]:
    raw_tokens[idx] = encode_frame(dino, idx).cpu()
    print(f'  frame {idx} done')
dino.cpu(); torch.cuda.empty_cache()

torch.manual_seed(FIXED_SEED)
fixed_noise = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
renderer    = make_renderer()

def fresh_lora():
    freeze_trellis(flow_model)
    lb = build_lora_blocks(rank=LORA_RANK, lora_alpha=LORA_RANK).to(DEVICE)
    return lb


# ═══════════════════════════════════════════════════════════════════════════════
# DIAG 5: ||B|| trajectory under eps=1e-16, wd=0  (2 epochs, 3 frames only)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DIAG 5: ||B|| trajectory under eps=1e-16, wd=0  (2 epochs, 3 frames)')
print('='*70)

lora_blocks = fresh_lora()
optimizer   = torch.optim.Adam(trainable_params(lora_blocks), lr=1e-4,
                               eps=1e-16, weight_decay=0.0)

diag_frames = [1, 75, 150]
b_traj = {blk_i: [] for blk_i in range(24)}  # step → ||B_i||
step_labels = []
flow_model.train()

for epoch in range(1, 3):
    for fi in diag_frames:
        K_hat, K_pool = blend_v2c(raw_tokens, fi)
        cond_gl = K_hat.unsqueeze(0)
        ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)
        optimizer.zero_grad()

        x_prefix = denoise_prefix_nograd(flow_model, ns, cond_gl, K_pool, lora_blocks)

        # first pass (no grad) → slat_val → decode
        with torch.no_grad():
            t, t_prev = T_PAIRS[-1]
            t_ten = torch.tensor([1000.*t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pool, lora_blocks, ALPHA):
                v = flow_model(x_prefix, t_ten, cond_gl)
            x0_val = x_prefix.replace(x_prefix.feats - (t - t_prev) * v.feats)

        slat_val = normalize_slat(x0_val)
        color, slat_leaf = decode_and_render(pipeline, slat_val, renderer, diag=False, device=DEVICE)
        gt        = load_gt(fi)
        task_loss = F.mse_loss(color, gt)
        (task_loss * LOSS_SCALE).backward()

        grad_slat = slat_leaf.grad.detach()   # keep at LOSS_SCALE

        x0_raw = denoise_last_with_grad(flow_model, x_prefix, cond_gl, K_pool, lora_blocks)
        slat   = normalize_slat(x0_raw)
        torch.autograd.backward(slat.feats, grad_slat)

        for p in trainable_params(lora_blocks):
            if p.grad is not None:
                p.grad.div_(LOSS_SCALE)

        has_bad = any(not torch.isfinite(p.grad).all()
                      for p in trainable_params(lora_blocks) if p.grad is not None)
        if has_bad:
            print(f'  e{epoch} f{fi:03d}: inf/nan in grads, skipping')
            optimizer.zero_grad()
        else:
            optimizer.step()

        for blk_i, blk in enumerate(lora_blocks):
            b_traj[blk_i].append(blk.lora_q.B.float().norm().item())
        step_labels.append(f'e{epoch}f{fi}')

        b_mean = np.mean([b_traj[i][-1] for i in range(24)])
        print(f'  e{epoch} f{fi:03d}  task={task_loss.item():.5f}  '
              f'||B||_mean={b_mean:.6f}  loss={task_loss.item():.5f}')

        del color, slat_leaf, gt, task_loss, x0_raw, slat, grad_slat
        del x_prefix, x0_val, slat_val, cond_gl, K_pool, K_hat
        gc.collect(); torch.cuda.empty_cache()

# Plot Diag 5
fig, ax = plt.subplots(figsize=(10, 4))
steps = list(range(len(step_labels)))
for blk_i in range(24):
    ax.plot(steps, b_traj[blk_i], alpha=0.4, linewidth=0.8)
mean_traj = [np.mean([b_traj[i][s] for i in range(24)]) for s in steps]
ax.plot(steps, mean_traj, 'k-', linewidth=2, label='mean across 24 blocks')
ax.set_xticks(steps); ax.set_xticklabels(step_labels, rotation=45, fontsize=8)
ax.set_xlabel('Training step'); ax.set_ylabel('||B|| (Frobenius norm)')
ax.set_title('Diag 5: ||B|| trajectory (eps=1e-16, wd=0, 2 epochs × 3 frames)')
ax.legend(); fig.tight_layout()
fig.savefig(_HERE / 'diag5_b_trajectory.png', dpi=150)
plt.close(fig)
print(f'[DIAG 5] saved → diag5_b_trajectory.png')
print(f'  final ||B||_mean = {mean_traj[-1]:.6f}')
if mean_traj[-1] > 1e-6:
    print('  ✓ B IS MOVING — eps fix works')
else:
    print('  ✗ B still flat — deeper problem, run diag 1 and 2')


# ═══════════════════════════════════════════════════════════════════════════════
# DIAG 1: Per-block gradient norm (one backward, all 24 blocks hooked)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DIAG 1: Per-block gradient norm (log scale)')
print('='*70)

lora_blocks = fresh_lora()
flow_model.train()

# Hook each transformer block's output SparseTensor.feats
block_feats   = {}
fwd_hooks     = []

for blk_i, block in enumerate(flow_model.blocks):
    def make_hook(idx):
        def hook(module, inp, out):
            # only capture during the grad-enabled last step; skip no-grad prefix
            if hasattr(out, 'feats') and out.feats is not None and out.feats.requires_grad:
                out.feats.retain_grad()
                block_feats[idx] = out.feats
        return hook
    fwd_hooks.append(block.register_forward_hook(make_hook(blk_i)))

fi = 75
K_hat, K_pool = blend_v2c(raw_tokens, fi)
cond_gl = K_hat.unsqueeze(0)
ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)

x_prefix = denoise_prefix_nograd(flow_model, ns, cond_gl, K_pool, lora_blocks)
x0_raw   = denoise_last_with_grad(flow_model, x_prefix, cond_gl, K_pool, lora_blocks)
slat     = normalize_slat(x0_raw)

# Use unit random gradient to measure pure attenuation (no loss bias)
fake_grad = torch.ones_like(slat.feats) / slat.feats.numel() ** 0.5
torch.autograd.backward(slat.feats, fake_grad * LOSS_SCALE)

for h in fwd_hooks:
    h.remove()

block_grad_norms = {}
for blk_i, feats in block_feats.items():
    if feats.grad is not None:
        block_grad_norms[blk_i] = feats.grad.float().norm().item()
    else:
        block_grad_norms[blk_i] = 0.0

for blk_i, norm in sorted(block_grad_norms.items()):
    print(f'  block {blk_i:02d}  ||grad||={norm:.4e}')

fig, ax = plt.subplots(figsize=(10, 4))
idxs  = sorted(block_grad_norms.keys())
norms = [block_grad_norms[i] for i in idxs]
ax.semilogy(idxs, [max(n, 1e-20) for n in norms], 'o-', color='steelblue')
ax.set_xlabel('Block index (0=earliest, 23=latest)')
ax.set_ylabel('||grad|| (log scale)')
ax.set_title('Diag 1: Gradient norm at each transformer block output')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(_HERE / 'diag1_block_grad_norm.png', dpi=150)
plt.close(fig)
print(f'[DIAG 1] saved → diag1_block_grad_norm.png')

del x_prefix, x0_raw, slat, block_feats, cond_gl, K_pool, K_hat
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# DIAG 2: SKIPPED — OOM
# Full 25-step backward graph requires ~25 × 24-block attention graphs ≫ 44GB.
# Not needed: Diag 5 confirmed eps fix works; Diag 1 shows healthy per-block grads.
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DIAG 2: SKIPPED (OOM: 25-step full graph > 44GB; not needed after Diag 5)')
print('='*70)
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# DIAG 3: B.grad per block for frames 1, 75, 150 (frame variation)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DIAG 3: B.grad per block — frame variation (f001, f075, f150)')
print('='*70)

frame_b_grads = {}

for fi in [1, 75, 150]:
    lora_blocks = fresh_lora()
    flow_model.train()

    K_hat, K_pool = blend_v2c(raw_tokens, fi)
    cond_gl = K_hat.unsqueeze(0)
    ns = sp.SparseTensor(feats=fixed_noise.clone(), coords=coords)

    x_prefix = denoise_prefix_nograd(flow_model, ns, cond_gl, K_pool, lora_blocks)

    with torch.no_grad():
        t, t_prev = T_PAIRS[-1]
        t_ten = torch.tensor([1000.*t], device=DEVICE, dtype=torch.float32)
        with dual_path_ctx_v2(flow_model, K_pool, lora_blocks, ALPHA):
            v = flow_model(x_prefix, t_ten, cond_gl)
        x0_val = x_prefix.replace(x_prefix.feats - (t - t_prev) * v.feats)

    slat_val = normalize_slat(x0_val)
    color, slat_leaf = decode_and_render(pipeline, slat_val, renderer, diag=False, device=DEVICE)
    gt        = load_gt(fi)
    task_loss = F.mse_loss(color, gt)
    (task_loss * LOSS_SCALE).backward()

    grad_slat = slat_leaf.grad.detach()

    x0_raw = denoise_last_with_grad(flow_model, x_prefix, cond_gl, K_pool, lora_blocks)
    slat   = normalize_slat(x0_raw)
    torch.autograd.backward(slat.feats, grad_slat)

    for p in trainable_params(lora_blocks):
        if p.grad is not None:
            p.grad.div_(LOSS_SCALE)

    grads = lora_b_grads(lora_blocks)
    frame_b_grads[fi] = grads
    print(f'  f{fi:03d}  B.grad norms: min={min(grads):.3e}  max={max(grads):.3e}')

    del color, slat_leaf, gt, task_loss, x0_raw, slat, grad_slat
    del x_prefix, x0_val, slat_val, cond_gl, K_pool, K_hat
    gc.collect(); torch.cuda.empty_cache()

fig, ax = plt.subplots(figsize=(10, 4))
colors = {1: 'steelblue', 75: 'darkorange', 150: 'green'}
for fi in [1, 75, 150]:
    ax.plot(range(24), frame_b_grads[fi], 'o-', color=colors[fi],
            label=f'frame {fi}', alpha=0.8)
ax.set_xlabel('Block index'); ax.set_ylabel('||B.grad|| (lora_q)')
ax.set_yscale('symlog', linthresh=1e-16)
ax.set_title('Diag 3: B gradient per block — frame variation')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(_HERE / 'diag3_frame_variation.png', dpi=150)
plt.close(fig)
print(f'[DIAG 3] saved → diag3_frame_variation.png')

# Check if lines differ
all_g = np.array([frame_b_grads[fi] for fi in [1, 75, 150]])
variation = np.std(all_g, axis=0).mean()
mean_mag  = np.abs(all_g).mean()
print(f'  Mean variation across frames: {variation:.3e}  mean magnitude: {mean_mag:.3e}')
if variation > 0.1 * mean_mag:
    print('  ✓ Gradients DIFFER across frames — signal carries frame content')
else:
    print('  ✗ Gradients near-IDENTICAL across frames — content-free signal')


# ═══════════════════════════════════════════════════════════════════════════════
# DIAG 4: Adam effective step size at eps=1e-8 vs eps=1e-16 (analytical)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DIAG 4: Adam effective step size at eps=1e-8 vs eps=1e-16')
print('='*70)

# Simulate Adam moment evolution over N steps for a given constant gradient g
def adam_step_sizes(g, n_steps=50, beta1=0.9, beta2=0.999, eps_val=1e-8, lr=1e-4):
    m1, m2 = 0.0, 0.0
    sizes = []
    for t in range(1, n_steps+1):
        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = beta2 * m2 + (1 - beta2) * g**2
        m1h = m1 / (1 - beta1**t)
        m2h = m2 / (1 - beta2**t)
        step = lr * m1h / (m2h**0.5 + eps_val)
        sizes.append(abs(step))
    return sizes

# Use observed gradient magnitude range
g_vals = [1e-15, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6]
steps  = list(range(1, 51))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, eps_val, eps_label in zip(axes, [1e-8, 1e-16], ['eps=1e-8 (default)', 'eps=1e-16 (fix)']):
    for g in g_vals:
        sizes = adam_step_sizes(g, eps_val=eps_val)
        ax.semilogy(steps, sizes, label=f'g={g:.0e}')
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Adam step size')
    ax.set_title(f'Adam effective step ({eps_label})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle('Diag 4: Adam step size vs gradient magnitude\n'
             'With eps=1e-8, tiny gradients are vetoed. eps=1e-16 restores scale invariance.')
fig.tight_layout()
fig.savefig(_HERE / 'diag4_adam_step_size.png', dpi=150)
plt.close(fig)
print(f'[DIAG 4] saved → diag4_adam_step_size.png')

# Print comparison at observed gradient magnitude
g_obs = 1.457e-12
for eps_val in [1e-8, 1e-16]:
    sizes = adam_step_sizes(g_obs, eps_val=eps_val, n_steps=10)
    print(f'  eps={eps_val:.0e}: step size at g={g_obs:.2e} → {sizes[-1]:.3e}  (ratio to lr: {sizes[-1]/1e-4:.3e})')

print('\n[DONE] All 5 diagnostics complete.')
print(f'Plots saved to: {_HERE}/')
