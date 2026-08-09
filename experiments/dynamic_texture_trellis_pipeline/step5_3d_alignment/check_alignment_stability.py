"""
Item 4 sanity check: does the voxel_to_token assignment change when you
use a blended 7-frame K_hat vs the single-frame K_hat used in training?

Camera fixed, mesh fixed → spatial layout should be frame-invariant.
If < ~5% of voxels change their assigned token, single-frame is fine.
If > ~20%, we need to use blended K_hat for alignment too.

Run from the step5_3d_alignment directory:
  cd .../step5_3d_alignment
  SPCONV_ALGO=native ATTN_BACKEND=xformers python check_alignment_stability.py
"""

import os, sys
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']      = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
from pathlib import Path

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

from step1_input_prep.input_prep       import load_frame, get_window_indices
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step4_mcfm.mcfm                   import run_mcfm
from step5_3d_alignment.alignment      import extract_alignment

DEVICE     = torch.device('cuda')
PRETRAINED = 'microsoft/TRELLIS-image-large'
NOISE_SEED = 42
K          = 3
FRAME_T    = 75

GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')


def main():
    print(f'=== Alignment stability check  frame={FRAME_T}  k={K} ===\n')

    # ── Load pipeline ──────────────────────────────────────────────────────────
    print('Loading TRELLIS...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Sample voxel structure (same seed as training) ─────────────────────────
    print('Sampling voxel structure...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox = {N_vox}\n')

    # ── DINOv2 ────────────────────────────────────────────────────────────────
    print('Loading DINOv2...')
    dino = load_dino(DEVICE)

    # ── Encode frames ─────────────────────────────────────────────────────────
    win_idx = get_window_indices(FRAME_T, K)   # [72,73,74,75,76,77,78]
    print(f'Window: {win_idx}')
    frames  = [load_frame(i) for i in win_idx]
    tokens  = encode_frames(win_idx, frames, dino, DEVICE)
    del dino
    torch.cuda.empty_cache()

    # ── Noise SparseTensor (same as training) ─────────────────────────────────
    torch.manual_seed(NOISE_SEED)
    sx = sp.SparseTensor(
        feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
        coords=coords,
    )

    # ── ALIGNMENT A: single-frame K_hat (current training code) ───────────────
    print('\n--- Alignment A: single-frame K_hat (window=[75] only) ---')
    tok_single = {FRAME_T: tokens[FRAME_T]}
    K_hat_single, _ = run_mcfm('v2', tok_single, [FRAME_T], FRAME_T,
                                torch.ones(1, device=DEVICE))
    cond_single = K_hat_single.unsqueeze(0)
    with torch.no_grad():
        _, _, v2t_single = extract_alignment(flow_model, sx, cond_single, DEVICE)
    print(f'  N_vox_ca = {v2t_single.shape[0]}')
    print(f'  Unique tokens assigned: {v2t_single.unique().shape[0]}')

    # ── ALIGNMENT B: blended 7-frame K_hat ────────────────────────────────────
    print('\n--- Alignment B: blended 7-frame K_hat (window=[72..78]) ---')
    lambda_vec = torch.full((len(win_idx),), 1.0 / len(win_idx), device=DEVICE)
    K_hat_blend, _ = run_mcfm('v2', tokens, win_idx, FRAME_T, lambda_vec)
    cond_blend  = K_hat_blend.unsqueeze(0)
    with torch.no_grad():
        _, _, v2t_blend = extract_alignment(flow_model, sx, cond_blend, DEVICE)
    print(f'  N_vox_ca = {v2t_blend.shape[0]}')
    print(f'  Unique tokens assigned: {v2t_blend.unique().shape[0]}')

    # ── Compare ───────────────────────────────────────────────────────────────
    print('\n--- Comparison ---')
    assert v2t_single.shape == v2t_blend.shape, 'voxel count mismatch'
    n_vox_ca   = v2t_single.shape[0]
    n_changed  = (v2t_single != v2t_blend).sum().item()
    pct        = 100.0 * n_changed / n_vox_ca

    print(f'  N_vox_ca          : {n_vox_ca}')
    print(f'  Voxels that change: {n_changed}  ({pct:.1f}%)')
    print(f'  Voxels unchanged  : {n_vox_ca - n_changed}  ({100-pct:.1f}%)')

    if pct < 5.0:
        print(f'\n  RESULT: STABLE ({pct:.1f}% < 5%). Single-frame alignment is fine.')
        print('  Paper note: "alignment is frame-invariant under fixed camera and mesh;')
        print('  center-frame conditioning suffices (< 5% voxel reassignment)."')
    elif pct < 20.0:
        print(f'\n  RESULT: MODERATE DRIFT ({pct:.1f}%). Worth noting but not critical.')
    else:
        print(f'\n  RESULT: HIGH DRIFT ({pct:.1f}%). Consider switching to blended K_hat for alignment.')

    # ── Token-level analysis of changed voxels ─────────────────────────────────
    changed_mask = v2t_single != v2t_blend
    if changed_mask.any():
        old_tokens = v2t_single[changed_mask]
        new_tokens = v2t_blend[changed_mask]

        # Are changed voxels moving to nearby patch positions?
        # Token index → patch row/col (37×37 grid, offset by 5 for CLS+REG)
        def tok_to_rc(t):
            p = t - 5   # patch index 0-based
            return p // 37, p % 37

        old_r, old_c = tok_to_rc(old_tokens)
        new_r, new_c = tok_to_rc(new_tokens)
        dist = ((old_r - new_r).float()**2 + (old_c - new_c).float()**2).sqrt()
        print(f'\n  For changed voxels, patch-grid displacement:')
        print(f'    mean  = {dist.mean().item():.2f} patches')
        print(f'    median= {dist.median().item():.2f} patches')
        print(f'    max   = {dist.max().item():.2f} patches')


if __name__ == '__main__':
    main()
