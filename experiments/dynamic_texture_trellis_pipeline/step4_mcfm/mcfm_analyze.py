"""
MCFM attention weight extractors — analysis only.

Same math as mcfm_v2 / mcfm_v3 in mcfm.py, but also return per-position
attention weights so we can analyze what TRELLIS naturally attends to.

mcfm.py is NOT modified — fully backward compatible.

Outputs
-------
mcfm_v2_attn  → (K_hat, K_hat, attn_weights)
    attn_weights : (N_TOKENS, n_window)  — softmax prob per spatial position per frame

mcfm_v3_attn  → (K_hat, K_hat, frame_attn)
    frame_attn   : (N_TOKENS, n_window)  — attn mass per spatial position summed over frame's token bank
"""

import torch
from .mcfm import _get_tokens, _stack_window, _SCALE, DINO_DIM


def mcfm_v2_attn(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-position temporal attention (same as mcfm_v2).
    Additionally returns attn_weights (N_TOKENS, n_window).
    """
    all_tokens = _stack_window(tokens, window_indices)   # (n_window, N_TOKENS, 1024)
    query_t    = _get_tokens(tokens, frame_idx_t)        # (N_TOKENS, 1024)

    Q = query_t.unsqueeze(1)                             # (N_TOKENS, 1, 1024)
    K = all_tokens.permute(1, 0, 2)                      # (N_TOKENS, n_window, 1024)
    V = K

    scores  = torch.bmm(Q, K.transpose(1, 2)) * _SCALE  # (N_TOKENS, 1, n_window)
    attn    = torch.softmax(scores, dim=-1)               # (N_TOKENS, 1, n_window)
    blended = torch.bmm(attn, V).squeeze(1)              # (N_TOKENS, 1024)

    return blended, blended, attn.squeeze(1)              # attn: (N_TOKENS, n_window)


def mcfm_v3_attn(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint temporal+spatial attention (same as mcfm_v3).
    Additionally returns frame_attn (N_TOKENS, n_window) — attention mass
    summed over each frame's token bank per query position.
    """
    all_tokens = _stack_window(tokens, window_indices)   # (n_window, N_TOKENS, 1024)
    n_frames   = len(window_indices)
    n_tokens   = all_tokens.shape[1]
    pool       = all_tokens.view(-1, DINO_DIM)           # (n_window*N_TOKENS, 1024)
    query_t    = _get_tokens(tokens, frame_idx_t)        # (N_TOKENS, 1024)

    scores     = torch.mm(query_t, pool.T) * _SCALE      # (N_TOKENS, n_window*N_TOKENS)
    attn       = torch.softmax(scores, dim=-1)            # (N_TOKENS, n_window*N_TOKENS)
    blended    = torch.mm(attn, pool)                     # (N_TOKENS, 1024)

    # Sum attention mass per frame → (N_TOKENS, n_window)
    frame_attn = attn.view(n_tokens, n_frames, n_tokens).sum(dim=-1)

    return blended, blended, frame_attn
