"""
Step 6.5 v2 — Dual-path cross-attention context manager.

BACKWARD COMPATIBLE: original ray_attention.py untouched.

PATH A  Spatial  (frozen):  q_base  @ K_hat^T    + soft_bias  -> out_sp
PATH B  Temporal (LoRA  ):  q_lora  @ K_pool^T               -> out_tp

out_final = to_out( out_sp + alpha * out_tp )

Key contracts:
  - lora_q  input : x_feats  (same input space as frozen to_q)
  - lora_kv input : K_pooled (same input space as frozen to_kv on path B)
  - enhance_bias  : (N_vox, N_tok) soft additive, PATH A ONLY, beta*bias_map
  - alpha=0 + B=zeros → out_final == out_sp == no-LoRA baseline  (GATE 1)
"""

import os
import torch
from contextlib import contextmanager

ATTN_CHUNK = int(os.environ.get('RAY_ATTN_CHUNK', '256'))


def _attn_chunked(q, k, v, scale, e_add=None, chunk=ATTN_CHUNK):
    """
    Memory-safe explicit attention.
    q : (N, H, hd)   k/v : (T, H, hd)
    e_add : (N, T) optional additive log-space bias (soft or hard -inf)
    Returns (N, H, hd) float32.
    """
    outs = []
    for s in range(0, q.shape[0], chunk):
        e   = min(s + chunk, q.shape[0])
        qc  = q[s:e].float()                              # (c, H, hd)
        A   = torch.einsum('nhd,mhd->nhm', qc, k.float()) * scale  # (c, H, T)
        if e_add is not None:
            A = A + e_add[s:e].unsqueeze(1).float()       # broadcast over heads
        A   = torch.softmax(A, dim=-1)
        out = torch.einsum('nhm,mhd->nhd', A, v.float())  # (c, H, hd)
        outs.append(out)
    return torch.cat(outs, dim=0)                         # (N, H, hd) float32


def _dual_path_fwd(module, x, context, K_pooled, lora_block, alpha, enhance_bias=None):
    """
    Replacement forward for one SparseMultiHeadAttention (cross type).

    module       : SparseMultiHeadAttention
    x            : SparseTensor, feats (N_vox, ch)
    context      : (1, T, cond_ch) — K_hat from MCFM (PATH A conditioning)
    K_pooled     : (T, cond_ch)   — temporal mean of individual frame tokens (PATH B)
    lora_block   : LoRABlock from lora_v2
    alpha        : nn.Parameter scalar
    enhance_bias : (N_vox, T) soft additive bias for PATH A, or None
    """
    H      = module.num_heads
    ch     = module.channels
    hd     = ch // H
    scale  = hd ** -0.5
    dt     = x.feats.dtype
    wdt    = module.to_q.weight.dtype

    x_feats = x.feats.to(wdt)                                     # (N, ch)
    ctx_2d  = (context.squeeze(0) if context.dim() == 3
               else context).to(wdt)                               # (T, cond_ch)
    Kp      = K_pooled.to(device=x_feats.device, dtype=wdt)       # (T, cond_ch)

    # ── PATH A: spatial, ALL frozen ──────────────────────────────────────────
    with torch.no_grad():
        q_base = module.to_q(x_feats)                              # (N, ch)
        kv_sp  = module.to_kv(ctx_2d).reshape(-1, 2, H, hd)       # (T, 2, H, hd)
    q_base_h = q_base.reshape(-1, H, hd)
    k_sp     = kv_sp[:, 0]
    v_sp     = kv_sp[:, 1]

    out_sp = _attn_chunked(q_base_h, k_sp, v_sp, scale,
                           e_add=enhance_bias)                     # (N, H, hd)

    # ── PATH B: temporal, LoRA on Q and KV ───────────────────────────────────
    # lora_q  input = x_feats  (same input space as frozen to_q)
    # lora_kv input = K_pooled (same input space as frozen to_kv)
    delta_q = lora_block.lora_q(x_feats)                          # (N, ch), has grad when B≠0
    q_lora  = (q_base.detach() + delta_q).reshape(-1, H, hd)      # (N, H, hd)

    with torch.no_grad():
        kv_frozen = module.to_kv(Kp)                               # (T, 2*ch), frozen
    delta_kv  = lora_block.lora_kv(Kp)                            # (T, 2*ch), grad when B≠0
    kv_tp     = (kv_frozen + delta_kv).reshape(-1, 2, H, hd)      # (T, 2, H, hd)
    k_tp      = kv_tp[:, 0]
    v_tp      = kv_tp[:, 1]

    out_tp = _attn_chunked(q_lora, k_tp, v_tp, scale,
                           e_add=None)                             # (N, H, hd) — NO bias on B

    # ── COMBINE at output level ───────────────────────────────────────────────
    out = (out_sp + alpha * out_tp).to(wdt)                        # (N, H, hd)
    out = out.reshape(-1, ch)                                      # (N, ch)
    out = module.to_out(out)                                       # (N, ch)
    return x.replace(out.to(dt))


@contextmanager
def dual_path_ctx_v2(flow_model, K_pooled, lora_blocks, alpha, enhance_bias=None):
    """
    Context manager: patches all 24 cross_attn blocks with dual-path forward.

    flow_model    : SLatFlowModel (TRELLIS, frozen)
    K_pooled      : (T, 1024) temporal mean of DINOv2 tokens for current frame
    lora_blocks   : nn.ModuleList[LoRABlock] from lora_v2.build_lora_blocks()
    alpha         : nn.Parameter scalar (learnable)
    enhance_bias  : (N_vox, 1374) soft additive bias, PATH A only, or None

    At alpha=0 + B=zeros: output == PATH A only == no-LoRA baseline.  (GATE 1)
    """
    saved    = {}
    lora_idx = 0

    for i, block in enumerate(flow_model.blocks):
        if not hasattr(block, 'cross_attn'):
            continue
        ca       = block.cross_attn
        saved[i] = ca.forward
        lb       = lora_blocks[lora_idx]
        lora_idx += 1

        def _make(mod, lb_):
            def _fwd(x, context=None):
                return _dual_path_fwd(mod, x, context, K_pooled, lb_, alpha, enhance_bias)
            return _fwd

        ca.forward = _make(ca, lb)

    try:
        yield
    finally:
        for i, block in enumerate(flow_model.blocks):
            if i in saved:
                block.cross_attn.forward = saved[i]


def build_soft_enhance_bias(voxel_to_token: torch.Tensor,
                             beta: float,
                             device: torch.device,
                             dtype=torch.float16) -> torch.Tensor:
    """
    Build soft additive enhancement bias for PATH A.

    voxel_to_token : (N_vox,) int64 — from step05b alignment matrix
    beta           : frozen scalar from step06 sweep winner
    Returns        : (N_vox, 1374) tensor, bias[v, assigned_tok] = beta, rest = 0
    """
    N_vox = voxel_to_token.shape[0]
    bias  = torch.zeros(N_vox, 1374, dtype=dtype, device=device)
    bias[torch.arange(N_vox, device=device), voxel_to_token.to(device)] = beta
    print(f'[ENHANCE BIAS] beta={beta}  shape={tuple(bias.shape)}'
          f'  nonzero={int((bias > 0).sum())}  dtype={dtype}')
    return bias
