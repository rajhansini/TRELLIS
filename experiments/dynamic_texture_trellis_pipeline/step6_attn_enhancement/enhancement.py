"""
Step 6: Local Attention Enhancement.

Build E (N_vox_ca, 1374) and patch G_L cross-attention to apply it before softmax.

E[v, assigned_token] = 0.0     (logit unchanged)
E[v, other_tokens]   = -inf    (exp(-inf)=0, zeroed out in softmax)

Apply as: softmax(A_raw + E, dim=-1)
→ 100% attention mass on the single assigned token per voxel.
"""

import torch
from contextlib import contextmanager
from trellis.modules import sparse as sp


def build_enhancement_matrix(
    voxel_to_token: torch.Tensor,  # (N_vox_ca,) long — from Step 5
    N_ctx: int = 1374,
) -> torch.Tensor:
    """
    Returns additive log-space mask E of shape (N_vox_ca, N_ctx).

    E[v, assigned_token] = 0.0    → logit unchanged, survives softmax
    E[v, all others]     = -inf   → exp(-inf) = 0, zeroed out in softmax

    Apply as: softmax(A_raw + E)  — routes 100% attention to assigned token.
    """
    device   = voxel_to_token.device
    N_vox_ca = voxel_to_token.shape[0]

    E = torch.full((N_vox_ca, N_ctx), float('-inf'), device=device, dtype=torch.float32)
    vox_idx = torch.arange(N_vox_ca, device=device)
    E[vox_idx, voxel_to_token] = 0.0

    return E  # (N_vox_ca, N_ctx)


def _enhanced_cross_attn_forward(module, x, context, E):
    """
    Manual cross-attention with E applied before softmax.
    Replaces xformers fused attention with explicit Q@K^T → ⊙E → softmax → @V.

    module  : SparseMultiHeadAttention (type="cross")
    x       : SparseTensor, feats (N_vox_ca, channels)  — post norm2
    context : tensor (1, N_ctx, cond_channels) or (N_ctx, cond_channels)
    E       : tensor (N_vox_ca, N_ctx)
    """
    num_heads = module.num_heads
    channels  = module.channels
    head_dim  = channels // num_heads
    dtype     = x.feats.dtype

    # Q from voxel features
    q = module.to_q(x.feats)                                 # (N_vox_ca, channels)
    q = q.reshape(-1, num_heads, head_dim)                   # (N_vox_ca, H, head_dim)

    # K, V from context
    ctx_2d = context.squeeze(0) if context.dim() == 3 else context   # (N_ctx, cond_channels)
    kv     = module.to_kv(ctx_2d)                            # (N_ctx, 2*channels)
    kv     = kv.reshape(ctx_2d.shape[0], 2, num_heads, head_dim)     # (N_ctx, 2, H, head_dim)
    k, v   = kv[:, 0], kv[:, 1]                             # (N_ctx, H, head_dim) each

    # Optional QK RMS norm
    if module.qk_rms_norm:
        q = module.q_rms_norm(q)
        k = module.k_rms_norm(k)

    # Raw attention logits: (N_vox_ca, H, N_ctx)
    scale  = head_dim ** -0.5
    A_raw  = torch.einsum('nhd,mhd->nhm', q.float(), k.float()) * scale

    # Apply E as additive mask across all heads: (N_vox_ca, 1, N_ctx) broadcast
    # E[v, assigned]=0, E[v, others]=-inf → non-assigned tokens zeroed out in softmax
    E_dev  = E.to(device=A_raw.device, dtype=A_raw.dtype)
    A_scaled = A_raw + E_dev.unsqueeze(1)                    # (N_vox_ca, H, N_ctx)

    # Softmax + weighted sum
    A_probs = torch.softmax(A_scaled, dim=-1).to(dtype)      # (N_vox_ca, H, N_ctx)
    out     = torch.einsum('nhm,mhd->nhd', A_probs, v.float().to(dtype))  # (N_vox_ca, H, head_dim)
    out     = out.reshape(-1, channels)                      # (N_vox_ca, channels)
    out     = module.to_out(out)                             # (N_vox_ca, channels)

    return x.replace(out)


@contextmanager
def enhanced_attention_ctx(flow_model, E: torch.Tensor):
    """
    Context manager that patches all cross_attn blocks in G_L to apply E
    before softmax. Original forward is restored on exit.

    Usage:
        with enhanced_attention_ctx(flow_model, E):
            output = flow_model(sparse_x, t, cond)
    """
    saved = {}

    for i, block in enumerate(flow_model.blocks):
        if not hasattr(block, 'cross_attn'):
            continue
        ca = block.cross_attn
        saved[i] = ca.forward

        def _make_patched(mod, E_mat):
            def patched(x, context=None):
                return _enhanced_cross_attn_forward(mod, x, context, E_mat)
            return patched

        ca.forward = _make_patched(ca, E)

    try:
        yield
    finally:
        for i, block in enumerate(flow_model.blocks):
            if i in saved:
                block.cross_attn.forward = saved[i]
