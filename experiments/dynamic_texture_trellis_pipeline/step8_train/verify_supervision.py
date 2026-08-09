"""
Phase A: Supervision Sanity Check

For a given frame:
  1. Render the mesh via TRELLIS (get render + render mask from nvdiffrast)
  2. Segment the GT frame via rembg (get GT object mask)
  3. Compute intersection mask = render_mask AND gt_mask
  4. Compute MSE on full image vs masked-only image
  5. Save side-by-side visualization

Usage:
  python verify_supervision.py --frame 75
  python verify_supervision.py --frame 1 --frame 75 --frame 150
"""

import os, sys, argparse
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from trellis.renderers import MeshRenderer
from trellis.pipelines import TrellisImageTo3DPipeline

# ── paths (same as train.py) ──────────────────────────────────────────────────
GT_FRAMES_DIR = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                 '/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
LATENT_NPZ    = ('/net/projects/ranalab/rajhansini/TRELLIS/data'
                 '/dynamic_sequences/trellis_seq/frame_0001/latent.npz')
PRETRAINED    = 'microsoft/TRELLIS-image-large'
CKPT_DIR      = '../results/lora_ckpts'
RESULTS_DIR   = '../results/supervision_check'
RENDER_RES    = 518
DEVICE        = torch.device('cuda')

# Camera (same as train.py)
EXTRINSICS = torch.tensor([
    [ 1,  0,  0,  0],
    [ 0,  0,  1,  2],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1],
], dtype=torch.float32)
INTRINSICS = torch.tensor([
    [1444.4,    0,  259],
    [   0,  1444.4, 259],
    [   0,     0,    1],
], dtype=torch.float32)


def load_gt_frame(frame_idx: int) -> np.ndarray:
    path = os.path.join(GT_FRAMES_DIR, f'frame_{frame_idx:04d}.png')
    img = Image.open(path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    return np.array(img)                          # HxWx3 uint8


def get_gt_mask(gt_rgb: np.ndarray) -> np.ndarray:
    """Use rembg to segment the object in the GT frame. Returns HxW bool mask."""
    from rembg import remove
    img_pil = Image.fromarray(gt_rgb)
    result  = remove(img_pil)                     # RGBA output
    alpha   = np.array(result)[:, :, 3]          # alpha channel
    return alpha > 128                            # HxW bool


def render_frame(pipeline, frame_idx: int, renderer: MeshRenderer):
    """Decode SLaT at frame_idx and render. Returns (rgb HxWx3 float, mask HxW bool)."""
    import trellis.modules.sparse as sp

    data = np.load(LATENT_NPZ)
    coords = torch.from_numpy(data['coords']).to(DEVICE)
    feats  = torch.from_numpy(data['feats']).float().to(DEVICE)

    slat = sp.SparseTensor(feats=feats, coords=coords)

    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]

    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    color = result['color'][0].permute(1, 2, 0).cpu().numpy()   # HxWx3 float [0,1]
    mask  = result['mask'][0, 0].cpu().numpy() > 0.5            # HxW bool
    return color, mask


def visualize(frame_idx: int, gt_rgb: np.ndarray, render_rgb: np.ndarray,
              gt_mask: np.ndarray, render_mask: np.ndarray,
              inter_mask: np.ndarray, out_path: str):
    gt_f   = gt_rgb.astype(np.float32) / 255.0
    ren_f  = render_rgb.clip(0, 1)

    mse_full   = ((gt_f - ren_f) ** 2).mean(axis=2)
    mse_masked = np.where(inter_mask, ((gt_f - ren_f) ** 2).mean(axis=2), np.nan)

    mse_full_val   = float(((gt_f - ren_f) ** 2).mean())
    mse_masked_val = float(np.nanmean(mse_masked)) if inter_mask.any() else float('nan')

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(f'Frame {frame_idx:04d}   '
                 f'MSE full={mse_full_val:.5f}  '
                 f'MSE masked={mse_masked_val:.5f}  '
                 f'overlap={inter_mask.mean()*100:.1f}%',
                 fontsize=13)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.05)

    def show(ax, img, title, cmap=None, vmin=None, vmax=None):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    show(fig.add_subplot(gs[0, 0]), gt_rgb,          'GT frame')
    show(fig.add_subplot(gs[0, 1]), ren_f,            'TRELLIS render')
    show(fig.add_subplot(gs[0, 2]), gt_mask,          'GT mask (rembg)',    cmap='gray')
    show(fig.add_subplot(gs[0, 3]), render_mask,      'Render mask (nvdiffrast)', cmap='gray')

    # GT masked by intersection
    gt_masked  = gt_f * inter_mask[:, :, None]
    ren_masked = ren_f * inter_mask[:, :, None]
    show(fig.add_subplot(gs[1, 0]), gt_masked,        'GT × intersection')
    show(fig.add_subplot(gs[1, 1]), ren_masked,        'Render × intersection')
    show(fig.add_subplot(gs[1, 2]), inter_mask,        'Intersection mask', cmap='gray')
    show(fig.add_subplot(gs[1, 3]), mse_full,
         f'Per-pixel MSE (full)\nmax={mse_full.max():.4f}', cmap='hot', vmin=0)

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    print(f'  MSE full={mse_full_val:.5f}  MSE masked={mse_masked_val:.5f}  '
          f'overlap={inter_mask.mean()*100:.1f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, action='append', default=None)
    args = parser.parse_args()
    frames = args.frame if args.frame else [1, 75, 150]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading pipeline...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.cuda()

    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )

    for frame_idx in frames:
        print(f'\n--- Frame {frame_idx:04d} ---')
        gt_rgb = load_gt_frame(frame_idx)

        print('  Segmenting GT with rembg...')
        gt_mask = get_gt_mask(gt_rgb)

        print('  Rendering TRELLIS mesh...')
        render_rgb, render_mask = render_frame(pipeline, frame_idx, renderer)

        inter_mask = gt_mask & render_mask

        out_path = os.path.join(RESULTS_DIR, f'supervision_frame_{frame_idx:04d}.png')
        visualize(frame_idx, gt_rgb, render_rgb, gt_mask, render_mask, inter_mask, out_path)


if __name__ == '__main__':
    main()
