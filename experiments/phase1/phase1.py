"""
Phase 1 — Diagnose Cross-Attention Shift
=========================================
Two diagnostics run together:

A) Token-delta analysis (CPU, fast):
   For each frame t:  T_t = lerp(T_0, T_150, alpha)
   delta[t] = ||T_t - T_0||_2  per patch token  → 37x37 heatmap
   error[t] = ||trellis_t - gt_t||  per pixel    → resized to 37x37
   Pearson correlation(delta, error) per frame.

B) Cross-attention maps (GPU, selected frames):
   Hook Q from cross_attn.to_q and K from cross_attn.to_kv
   at every diffusion step. Compute softmax(QK^T/sqrt(d)) per head,
   average heads + steps → per-voxel attention over 1374 image tokens.
   Project attention back to image space (37x37) and compare to error map.

Output: experiments/results/phase1/
  token_delta.mp4      — [gt | interp | token-delta | error-map] per frame
  attn_maps.mp4        — [gt | error | attn-map] for selected frames (--attn)
  correlation.csv      — per-frame Pearson r between token-delta and error
  correlation.png      — plot
  run.log
"""

import os, sys
os.environ['SPCONV_ALGO'] = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import csv
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
PHASE0_DIR     = REPO_ROOT / 'experiments' / 'results' / 'phase0'
OUT_DIR        = REPO_ROOT / 'experiments' / 'results' / 'phase1'
FRAMES_150_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                      '/mvadaptorresults/trellis_150_frames')
SLAT_SEQ_DIR   = REPO_ROOT / 'data' / 'dynamic_sequences' / 'trellis_seq'
PRETRAINED     = 'microsoft/TRELLIS-image-large'

N_FRAMES   = 150
PATCH_H    = 37          # DINOv2 518x518 / patch 14 = 37
PATCH_W    = 37
N_PATCHES  = PATCH_H * PATCH_W   # 1369
# token layout: [CLS, reg0, reg1, reg2, reg3, patch_0, ..., patch_1368]
PATCH_START = 5          # skip CLS + 4 register tokens
RENDER_RES  = 512

# Frames to run attention hook extraction (subset, GPU-expensive)
ATTN_FRAMES = [1, 25, 50, 75, 100, 125, 150]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_phase0_frames():
    """Returns (trellis_imgs, gt_imgs) as float32 arrays [N, H, W, 3] in [0,1]."""
    frames_dir = PHASE0_DIR / 'frames'
    trellis, gt = [], []
    for t in range(1, N_FRAMES + 1):
        tp = frames_dir / f'trellis_{t:03d}.png'
        gp = frames_dir / f'gt_{t:03d}.png'
        trellis.append(np.array(Image.open(tp).convert('RGB')).astype(np.float32) / 255.)
        gt.append(np.array(Image.open(gp).convert('RGB').resize(
            (RENDER_RES, RENDER_RES), Image.LANCZOS)).astype(np.float32) / 255.)
    return np.stack(trellis), np.stack(gt)


def error_map(pred, gt):
    """L2 error per pixel, averaged over channels → (H,W) float32."""
    return np.sqrt(((pred - gt) ** 2).mean(axis=-1))


def colorize(arr, vmin=None, vmax=None):
    """Normalize and apply a hot colormap → (H,W,3) uint8."""
    import matplotlib.cm as cm
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = cm.hot(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def pearson(a, b):
    a, b = a.flatten(), b.flatten()
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = np.sqrt(((a - ma)**2).sum() * ((b - mb)**2).sum()) + 1e-10
    return float(num / den)


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Token delta analysis (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_token_delta(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_out = OUT_DIR / 'token_delta_frames'
    frames_out.mkdir(exist_ok=True)

    print('Loading tokens ...')
    data = np.load(PHASE0_DIR / 'tokens.npz')
    T_0   = data['T_0'].astype(np.float32)    # (1, 1374, 1024)
    T_150 = data['T_150'].astype(np.float32)

    print('Loading Phase 0 renders ...')
    trellis_imgs, gt_imgs = load_phase0_frames()
    err_global_max = None

    rows = []
    delta_maps, err_maps = [], []

    for t in range(1, N_FRAMES + 1):
        alpha = (t - 1) / (N_FRAMES - 1)
        T_t   = (1.0 - alpha) * T_0 + alpha * T_150   # (1, 1374, 1024)

        # Token delta: per-patch L2 norm of (T_t - T_0)
        diff    = (T_t - T_0)[0, PATCH_START:, :]   # (1369, 1024)
        delta   = np.linalg.norm(diff, axis=-1)      # (1369,)
        delta_m = delta.reshape(PATCH_H, PATCH_W)    # (37, 37)

        # Error map from Phase 0
        err   = error_map(trellis_imgs[t-1], gt_imgs[t-1])   # (512, 512)
        err_m = F.interpolate(
            torch.from_numpy(err).unsqueeze(0).unsqueeze(0),
            size=(PATCH_H, PATCH_W), mode='bilinear', align_corners=False
        )[0, 0].numpy()                                        # (37, 37)

        delta_maps.append(delta_m)
        err_maps.append(err_m)
        rows.append({'t': t, 'alpha': round(alpha, 4)})

    # global color scale
    d_max = max(m.max() for m in delta_maps)
    e_max = max(m.max() for m in err_maps)

    print('Computing correlation and saving frames ...')
    for i, t in enumerate(range(1, N_FRAMES + 1)):
        r = pearson(delta_maps[i], err_maps[i])
        rows[i]['pearson_r'] = round(r, 4)

        gt_rgb    = (gt_imgs[i] * 255).astype(np.uint8)
        pred_rgb  = (trellis_imgs[i] * 255).astype(np.uint8)

        # upscale heatmaps to RENDER_RES
        def up(m, vmax):
            col = colorize(m, vmin=0, vmax=vmax)   # (37,37,3)
            return np.array(Image.fromarray(col).resize((RENDER_RES, RENDER_RES), Image.NEAREST))

        delta_vis = up(delta_maps[i], d_max)
        err_vis   = up(err_maps[i],   e_max)

        panel = np.concatenate([gt_rgb, pred_rgb, delta_vis, err_vis], axis=1)
        Image.fromarray(panel).save(frames_out / f'frame_{t:03d}.png')

        if t % 25 == 0 or t == 1:
            print(f'  t={t:03d}  α={rows[i]["alpha"]:.3f}  r={r:.3f}')

    # CSV
    csv_path = OUT_DIR / 'correlation.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'alpha', 'pearson_r'])
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved: {csv_path}')

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ts = [r['t'] for r in rows]
        rs = [r['pearson_r'] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, rs, '-o', markersize=3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Frame t'); ax.set_ylabel('Pearson r (token-delta vs render-error)')
        ax.set_title('Phase 1: Token-delta ↔ Render-error correlation per frame')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'correlation.png', dpi=150)
        plt.close()
        print(f'Saved: {OUT_DIR}/correlation.png')
    except Exception as e:
        print(f'Plot failed: {e}')

    # Video: [gt | interp | token-delta | error-map]
    vid_path = OUT_DIR / 'token_delta.mp4'
    try:
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '15',
            '-i', str(frames_out / 'frame_%03d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid_path),
        ], check=True)
        print(f'Saved: {vid_path}')
    except Exception as e:
        print(f'Video failed: {e}')

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Cross-attention extraction (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttnHook:
    """
    Registers hooks on SparseMultiHeadAttention cross-attention forward to
    capture Q and K tensors. Computes attn = mean_heads(softmax(QK^T/sqrt(d))).
    Accumulates across diffusion steps then averages.
    """
    def __init__(self):
        self.q_list  = []   # list of (T_voxels, H, C) tensors per step
        self.k_list  = []
        self._hooks  = []

    def register(self, model):
        from trellis.modules.sparse.transformer.blocks import SparseTransformerCrossBlock
        for name, module in model.named_modules():
            if isinstance(module, SparseTransformerCrossBlock):
                h = module.cross_attn.register_forward_hook(self._hook_fn)
                self._hooks.append(h)

    def _hook_fn(self, module, inputs, output):
        # inputs[0] = q (SparseTensor or Tensor), inputs[1] = kv (dense [N,L,2,H,C])
        x, context = inputs[0], inputs[1]
        from trellis.modules import sparse as sp
        # Q
        q_raw = module._linear(module.to_q, x)
        q_raw = module._reshape_chs(q_raw, (module.num_heads, -1))
        q = q_raw.feats if isinstance(q_raw, sp.SparseTensor) else q_raw  # (T_q, H, C)
        # K
        kv_raw = module._linear(module.to_kv, context)
        kv_raw = module._fused_pre(kv_raw, num_fused=2)
        kv = kv_raw.feats if isinstance(kv_raw, sp.SparseTensor) else kv_raw
        if kv.dim() == 3:
            k = kv[:, 0, :]  # (T_kv, H, C)... actually (T_kv, 2, H, C)
        # kv shape: (N*L, 2, H, C) or (T_kv, 2, H, C)
        # After _fused_pre with num_fused=2, shape: (T_kv, 2, H, C)
        if kv.dim() == 4:
            k = kv[:, 0, :, :]   # (T_kv, H, C)
        else:
            return  # unexpected shape, skip

        self.q_list.append(q.detach().cpu().float())
        self.k_list.append(k.detach().cpu().float())

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_attn_map(self):
        """Average attention over layers and steps → (T_voxels, T_kv) per step."""
        if not self.q_list:
            return None
        # Each step has multiple layers; average within step then across steps
        # q_list length = n_steps * n_layers
        # k_list same; all k should have same T_kv = 1374
        q_all = torch.stack(self.q_list)   # (S, T_q, H, C)
        k_all = torch.stack(self.k_list)   # (S, T_kv, H, C)
        d = q_all.shape[-1]
        # (S, T_q, H, T_kv)
        attn = torch.einsum('sqhc,skhc->sqhk', q_all, k_all) / math.sqrt(d)
        attn = torch.softmax(attn, dim=-1)   # (S, T_q, H, T_kv)
        attn = attn.mean(dim=(0, 2))          # avg steps + heads → (T_q, T_kv)
        self.q_list.clear(); self.k_list.clear()
        return attn.numpy()


def run_attention_maps(frames=ATTN_FRAMES):
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.modules import sparse as sp

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attn_frames_dir = OUT_DIR / 'attn_frames'
    attn_frames_dir.mkdir(exist_ok=True)

    print('Loading pipeline for attention extraction ...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()
    device = pipeline.device

    data = np.load(PHASE0_DIR / 'tokens.npz')
    T_0   = torch.from_numpy(data['T_0'].astype(np.float32)).to(device)
    T_150 = torch.from_numpy(data['T_150'].astype(np.float32)).to(device)

    data_c = np.load(SLAT_SEQ_DIR / 'frame_0001' / 'latent.npz')
    coords = data_c['coords'].astype(np.int32)
    batch  = np.zeros((len(coords), 1), dtype=np.int32)
    coords4 = torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(device)

    # Load phase0 renders for overlay
    trellis_imgs, gt_imgs = load_phase0_frames()

    err_maps_all = [error_map(trellis_imgs[t-1], gt_imgs[t-1]) for t in frames]
    e_max = max(m.max() for m in err_maps_all)

    hook = CrossAttnHook()
    hook.register(pipeline.slat_flow_model)

    for i, t in enumerate(frames):
        alpha  = (t - 1) / (N_FRAMES - 1)
        T_t    = (1.0 - alpha) * T_0 + alpha * T_150
        cond_t = {'cond': T_t, 'neg_cond': torch.zeros_like(T_t)}

        torch.manual_seed(42)
        with torch.no_grad():
            _ = pipeline.sample_slat(cond_t, coords4, sampler_params={'steps': 25})

        attn = hook.compute_attn_map()   # (T_q, 1374) or None

        gt_rgb   = (gt_imgs[t-1] * 255).astype(np.uint8)
        err_vis  = colorize(err_maps_all[i], vmin=0, vmax=e_max)
        err_vis  = np.array(Image.fromarray(err_vis).resize((RENDER_RES, RENDER_RES), Image.NEAREST))

        if attn is not None:
            # Average over voxels → attention over image tokens → patch tokens
            attn_token = attn.mean(axis=0)            # (1374,)
            attn_patch = attn_token[PATCH_START:]     # (1369,)
            attn_map   = attn_patch.reshape(PATCH_H, PATCH_W)
            attn_vis   = colorize(attn_map)
            attn_vis   = np.array(Image.fromarray(attn_vis).resize((RENDER_RES, RENDER_RES), Image.NEAREST))
        else:
            attn_vis = np.zeros((RENDER_RES, RENDER_RES, 3), dtype=np.uint8)

        panel = np.concatenate([gt_rgb, err_vis, attn_vis], axis=1)
        Image.fromarray(panel).save(attn_frames_dir / f'frame_{t:03d}.png')
        print(f'  t={t:03d}  α={alpha:.3f}  attn_extracted={attn is not None}')

    hook.remove()

    # Save grid of key frames
    panels = [np.array(Image.open(attn_frames_dir / f'frame_{t:03d}.png')) for t in frames]
    grid   = np.concatenate(panels, axis=0)
    Image.fromarray(grid).save(OUT_DIR / 'attn_grid.png')
    print(f'Saved: {OUT_DIR}/attn_grid.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn', action='store_true',
                        help='Also extract cross-attention maps (GPU required)')
    parser.add_argument('--attn_frames', type=int, nargs='+', default=ATTN_FRAMES,
                        help='Frames to extract attention for')
    args = parser.parse_args()

    print('=== Phase 1A: Token-delta analysis ===')
    rows = run_token_delta(args)

    avg_r = np.mean([r['pearson_r'] for r in rows])
    mid_r = rows[74]['pearson_r']
    print(f'\nSummary:')
    print(f'  Mean Pearson r across 150 frames: {avg_r:.3f}')
    print(f'  r at mid-point (t=75):            {mid_r:.3f}')
    if avg_r > 0.4:
        print('  → Token delta CORRELATES with render error.')
        print('    Attention scaling / weighted interpolation may fix Phase 0.')
    else:
        print('  → Token delta does NOT correlate with render error.')
        print('    Problem is deeper — likely need Phase 2 (lightweight correction).')

    if args.attn:
        print('\n=== Phase 1B: Cross-attention map extraction ===')
        run_attention_maps(args.attn_frames)

    print(f'\nOutputs: {OUT_DIR}')


if __name__ == '__main__':
    main()
