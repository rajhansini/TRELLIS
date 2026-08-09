"""
Step 6.5 — Dual-path (Spatial + Temporal) attention context manager.

PATH A  Spatial  (frozen):  Q_base @ K̂^T         → out_spatial
PATH B  Temporal (LoRA):    Q_lora @ K_pooled^T  → out_temporal

out_final = to_out( out_spatial + α * out_temporal )

K_pooled = mean([tokens[idx]['tokens'] for idx in window_indices])
         = simple temporal pool of individual DINOv2 frame tokens (no MCFM blend)
"""

import os
import torch
from contextlib import contextmanager

ATTN_CHUNK = int(os.environ.get('RAY_ATTN_CHUNK', '256'))
ATTN_COMPUTE = os.environ.get('RAY_ATTN_DTYPE', 'float16').lower()
ATTN_DTYPE = torch.float16 if ATTN_COMPUTE in ('float16', 'fp16', 'half') else torch.float32


def _compute_attn_chunked(q_heads, k, v, scale, chunk_size=ATTN_CHUNK, e_add=None):
    """
    Memory-safe scaled dot-product attention over query chunks.
    q_heads: (N, H, hd), k/v: (T, H, hd), optional e_add: (N, T) additive log-space mask
    e_add[v, assigned]=0, e_add[v, others]=-inf → 100% attention on assigned token.
    """
    n = q_heads.shape[0]
    outs = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        q = q_heads[s:e]  # (c, H, hd)
        A = torch.einsum('nhd,mhd->nhm', q, k) * scale  # (c, H, T)
        if e_add is not None:
            A = A + e_add[s:e].unsqueeze(1)
        A = torch.softmax(A, dim=-1)
        out = torch.einsum('nhm,mhd->nhd', A, v)  # (c, H, hd)
        outs.append(out)
    return torch.cat(outs, dim=0)


def _compute_attn(q_heads, k, v, scale):
    """Scaled dot-product attention. All inputs float32. Returns (N, H, head_dim)."""
    return _compute_attn_chunked(q_heads, k, v, scale)


def _dual_path_forward(module, x, context, K_pooled, lora_block, alpha, E=None):
    """
    Replacement for SparseMultiHeadAttention.forward (cross type).

    module     : SparseMultiHeadAttention
    x          : SparseTensor, feats (N_vox_ca, channels)  — post norm2, in model dtype
    context    : tensor (1, T, cond_ch) — K̂ blended conditioning from STEP 4
    K_pooled   : tensor (T, cond_ch)   — temporal mean of individual frame tokens
    lora_block : LoRABlock
    alpha      : nn.Parameter scalar
    """
    H   = module.num_heads
    ch  = module.channels
    hd  = ch // H
    dt  = x.feats.dtype
    mdtype = module.to_q.weight.dtype

    x_feats = x.feats.to(mdtype)                             # (N, ch)
    ctx_2d  = (context.squeeze(0) if context.dim() == 3 else context).to(mdtype)  # (T, cond_ch)
    Kp      = K_pooled.to(device=x_feats.device, dtype=mdtype)  # (T, cond_ch)

    scale = hd ** -0.5

    # ── PATH A: Spatial — frozen to_q, frozen to_kv on K̂ ─────────────────────
    # Compute q_base once with no_grad (used by both paths)
    with torch.no_grad():
        q_base = module.to_q(x_feats)                                     # (N, ch) fp16, no grad
        kv_sp  = module.to_kv(ctx_2d).reshape(-1, 2, H, hd)
    q_base_h = q_base.reshape(-1, H, hd).to(dtype=ATTN_DTYPE)
    k_sp, v_sp = kv_sp[:,0].to(dtype=ATTN_DTYPE), kv_sp[:,1].to(dtype=ATTN_DTYPE)
    if E is not None:
        # Step 6 enhancement: boost assigned-token logits before softmax
        E_dev  = E.to(device=q_base_h.device, dtype=q_base_h.dtype)
        out_sp = _compute_attn_chunked(q_base_h, k_sp, v_sp, scale, e_add=E_dev)
    else:
        out_sp = _compute_attn(q_base_h, k_sp, v_sp, scale)  # (N, H, hd) float32

    # ── PATH B: Temporal — q_base + LoRA delta, LoRA KV on K_pooled ──────────
    # Reuse q_base; only the LoRA delta needs grad tracking
    lora_q  = lora_block.lora_q
    A_q     = lora_q.lora_A.to(mdtype)
    B_q     = lora_q.lora_B.to(mdtype)
    delta_q = (x_feats @ A_q.T) @ B_q.T * lora_q.scaling    # (N, ch), has grad
    q_lora  = (q_base.detach() + delta_q).reshape(-1, H, hd).to(dtype=ATTN_DTYPE)

    kv_tp   = lora_block.lora_kv(Kp).reshape(-1, 2, H, hd)
    k_tp, v_tp = kv_tp[:,0].to(dtype=ATTN_DTYPE), kv_tp[:,1].to(dtype=ATTN_DTYPE)
    out_tp  = _compute_attn(q_lora, k_tp, v_tp, scale)       # (N, H, hd) float32

    # ── COMBINE ───────────────────────────────────────────────────────────────
    out = (out_sp + alpha.float() * out_tp).to(mdtype)        # (N, H, hd)
    out = out.reshape(-1, ch)                                  # (N, ch)
    out = module.to_out(out)                                   # (N, ch)

    return x.replace(out.to(dt))


@contextmanager
def dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=None):
    """
    Context manager: patches all cross_attn blocks in G_L with dual-path forward.

    K_pooled    : (T, cond_ch) tensor — temporal mean of individual frame tokens
    lora_blocks : nn.ModuleList[LoRABlock] from insert_lora()
    alpha       : nn.Parameter scalar
    E           : (N_vox_ca, T) tensor or None — Step 6 enhancement matrix (optional)

    Usage:
        with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=E):
            output = flow_model(sparse_x, t, cond)   # cond = K̂ from STEP 4
    """
    saved   = {}
    lora_idx = 0

    for i, block in enumerate(flow_model.blocks):
        if not hasattr(block, 'cross_attn'):
            continue
        ca         = block.cross_attn
        saved[i]   = ca.forward
        lb         = lora_blocks[lora_idx]
        lora_idx  += 1

        def _make(mod, lora_b):
            def _fwd(x, context=None):
                return _dual_path_forward(mod, x, context, K_pooled, lora_b, alpha, E)
            return _fwd

        ca.forward = _make(ca, lb)

    try:
        yield
    finally:
        for i, block in enumerate(flow_model.blocks):
            if i in saved:
                block.cross_attn.forward = saved[i]
