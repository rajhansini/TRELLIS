"""
Step 5: 3D Semantic-Aware Alignment

Hooks into G_L's (SLatFlowModel) cross-attention layers during a single forward
pass to extract attention maps between voxels and conditioning tokens.

INPUT:
  flow_model   : SLatFlowModel (from pipeline.models['slat_flow_model'])
  sparse_x     : sp.SparseTensor  — voxel structure (coords fixed, feats = noise)
  cond         : tensor (1, 1374, 1024)  — blended K̂, V̂ from STEP 4
  device       : torch.device
  threshold    : float, forward-alignment threshold (default 0.1)

OUTPUT:
  attn_map     : tensor (N_vox_ca, 1374) — averaged cross-attention across all layers
                 NOTE: N_vox_ca != N_vox input. G_L's input_blocks downsample voxels
                 before the cross-attention blocks. Alignment is at the downsampled level.
  token_to_voxels : dict {token_j: [voxel_indices]}  (forward alignment, indices into N_vox_ca)
  voxel_to_token  : tensor (N_vox_ca,) int — one token per downsampled voxel
"""

import torch
from trellis.modules import sparse as sp


def _capture_cross_attn_maps(flow_model, sparse_x, cond, device):
    """
    Single G_L forward pass with K̂, V̂ as conditioning.
    Hooks all cross-attention blocks to capture attention maps.

    Returns: list of (num_voxels, N_ctx) tensors, one per cross-attn layer.
    """
    collected = []

    def _hook(module, input, output):
        with torch.no_grad():
            h_sp  = input[0]          # SparseTensor: feats (N_vox, channels)
            ctx   = input[1]          # (1, N_ctx, channels) or (N_ctx, channels)

            n_heads  = module.num_heads
            channels = module.channels
            head_dim = channels // n_heads

            # Q from voxel features
            q = module.to_q(h_sp.feats)                           # (N_vox, channels)
            q = q.reshape(-1, n_heads, head_dim)                  # (N_vox, heads, head_dim)

            # K from context — squeeze batch dim if present
            ctx_2d = ctx.squeeze(0) if ctx.dim() == 3 else ctx   # (N_ctx, channels)
            kv     = module.to_kv(ctx_2d)                         # (N_ctx, 2*channels)
            kv     = kv.reshape(kv.shape[0], 2, n_heads, head_dim)
            k      = kv[:, 0]                                     # (N_ctx, heads, head_dim)

            # Attention map: (N_vox, heads, N_ctx) → average heads → (N_vox, N_ctx)
            scale = head_dim ** -0.5
            attn  = torch.einsum('nhd,mhd->nhm', q.float(), k.float()) * scale
            attn  = torch.softmax(attn, dim=-1).mean(dim=1)       # (N_vox, N_ctx)
            collected.append(attn.cpu())

    # Register hooks on every cross-attention block in G_L
    hooks = []
    for block in flow_model.blocks:
        if hasattr(block, 'cross_attn'):
            hooks.append(block.cross_attn.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            t = torch.tensor([500.0], device=device)   # mid-timestep (out of 1000)
            flow_model(sparse_x, t, cond)
    finally:
        for h in hooks:
            h.remove()

    return collected


def extract_alignment(
    flow_model,
    sparse_x: 'sp.SparseTensor',
    cond: torch.Tensor,
    device: torch.device,
    threshold: float = 0.1,
):
    """
    Full Step 5: run G_L forward, capture cross-attn maps, build alignment.

    Returns:
      attn_map        : tensor (num_voxels, 1374)
      token_to_voxels : dict {int: list[int]}
      voxel_to_token  : tensor (num_voxels,) dtype=long
    """
    # ── Capture attention maps across all cross-attn layers ──────────────────
    layer_maps = _capture_cross_attn_maps(flow_model, sparse_x, cond, device)
    assert len(layer_maps) > 0, "No cross-attention layers found in flow_model.blocks"

    # Average across all layers: (num_voxels, N_ctx)
    attn_map = torch.stack(layer_maps, dim=0).mean(dim=0)   # (N_vox, 1374)

    N_vox, N_ctx = attn_map.shape

    # ── Forward alignment: tokens → voxels (threshold) ───────────────────────
    token_to_voxels = {}
    for j in range(N_ctx):
        aligned = torch.where(attn_map[:, j] > threshold)[0].tolist()
        token_to_voxels[j] = aligned

    # ── Reverse alignment: every voxel → one patch token (argmax over patches only) ──
    # Restrict to patch tokens (indices 5–1373), skipping CLS(0) and REG(1–4).
    # REG tokens absorb high-magnitude artifact activations and dominate the global
    # argmax, causing every voxel to map to REG with no spatial content.
    # TODO: future — assign voxels with weak patch attention to CLS as fallback.
    voxel_to_token = attn_map[:, 5:].argmax(dim=1) + 5   # (N_vox,) long, values in [5, 1373]

    return attn_map, token_to_voxels, voxel_to_token
