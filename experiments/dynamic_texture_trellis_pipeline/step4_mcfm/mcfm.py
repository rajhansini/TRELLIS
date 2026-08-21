"""
Step 4: Multi-Condition Fusion Module (MCFM)

INPUT:
  tokens        : dict {frame_idx: {"cls":(1,1024), "reg":(4,1024), "patches":(256,1024)}}
  window_indices: list[int]  2k+1 indices (from Step 1, may have boundary duplicates)
  frame_idx_t   : int        target frame index
  lambda_vec    : tensor (2k+1,) frame weights from Step 3

OUTPUT (all versions):
  K_hat : tensor (1374, 1024)  blended keys
  V_hat : tensor (1374, 1024)  blended values

Token layout per frame: [CLS(1) | REG(4) | patches(1369)] = 1374 tokens, 1024 dim
(518×518 input, patch_size=14 → 37×37=1369 patches)

Versions:
  v1  — Naive weighted average
  v2  — Temporal-then-spatial (per-position cross-frame attention)
  v2b — Spatial-then-temporal (self-attention then cross-frame)
  v3  — Joint temporal+spatial (full pool attention)
"""

import torch

N_TOKENS = 1374  # 1 CLS + 4 REG + 1369 patches (518x518 / patch_size=14 → 37x37=1369)
DINO_DIM = 1024
_SCALE   = DINO_DIM ** -0.5


def _get_tokens(tokens: dict, frame_idx: int) -> torch.Tensor:
    """Return full conditioning tokens for one frame → (1374, 1024)."""
    return tokens[frame_idx]['tokens']   # (1374, 1024)


def _stack_window(tokens: dict, window_indices: list) -> torch.Tensor:
    """Stack tokens for all window frames → (2k+1, 261, 1024).
    Boundary duplicates repeat the same tensor, which is correct."""
    return torch.stack([_get_tokens(tokens, idx) for idx in window_indices], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# V1: Naive weighted average
# ─────────────────────────────────────────────────────────────────────────────

def mcfm_v1(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K̂[j] = Σ λ_i * tokens[i][j]   for each position j
    Output: (261, 1024)
    """
    all_tokens = _stack_window(tokens, window_indices)   # (2k+1, 261, 1024)
    weights    = lambda_vec.view(-1, 1, 1)               # (2k+1, 1, 1)
    blended    = (weights * all_tokens).sum(dim=0)       # (261, 1024)
    return blended, blended


# ─────────────────────────────────────────────────────────────────────────────
# V2: Temporal THEN Spatial
# ─────────────────────────────────────────────────────────────────────────────

def mcfm_v2(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per token position j: frame_t's token attends over all frames' position j.
    Temporal blending happens position-by-position.
    Output: (261, 1024)
    """
    all_tokens = _stack_window(tokens, window_indices)   # (2k+1, 261, 1024)
    query_t    = _get_tokens(tokens, frame_idx_t)        # (261, 1024)

    # Reshape for batched matmul over 261 positions
    Q = query_t.unsqueeze(1)                      # (261, 1, 1024)
    K = all_tokens.permute(1, 0, 2)               # (261, 2k+1, 1024)
    V = K

    scores  = torch.bmm(Q, K.transpose(1, 2)) * _SCALE   # (261, 1, 2k+1)
    attn    = torch.softmax(scores, dim=-1)                # (261, 1, 2k+1)
    blended = torch.bmm(attn, V).squeeze(1)               # (261, 1024)
    return blended, blended


# ─────────────────────────────────────────────────────────────────────────────
# V2b: Spatial THEN Temporal
# ─────────────────────────────────────────────────────────────────────────────

def mcfm_v2b(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1: Self-attention within frame_t to get spatially-refined tokens.
    Stage 2: Refined tokens attend over all frames (temporal blend).
    Output: (261, 1024)
    """
    frame_t = _get_tokens(tokens, frame_idx_t)   # (261, 1024)

    # Stage 1: spatial self-attention on frame_t
    Q = frame_t.unsqueeze(0)                                    # (1, 261, 1024)
    scores  = torch.bmm(Q, Q.transpose(1, 2)) * _SCALE         # (1, 261, 261)
    attn    = torch.softmax(scores, dim=-1)                     # (1, 261, 261)
    refined = torch.bmm(attn, Q).squeeze(0)                    # (261, 1024)

    # Stage 2: temporal cross-frame attention using refined as query
    all_tokens = _stack_window(tokens, window_indices)          # (2k+1, 261, 1024)
    Q2 = refined.unsqueeze(1)                                   # (261, 1, 1024)
    K2 = all_tokens.permute(1, 0, 2)                           # (261, 2k+1, 1024)
    V2 = K2

    scores2  = torch.bmm(Q2, K2.transpose(1, 2)) * _SCALE     # (261, 1, 2k+1)
    attn2    = torch.softmax(scores2, dim=-1)                   # (261, 1, 2k+1)
    blended  = torch.bmm(attn2, V2).squeeze(1)                 # (261, 1024)
    return blended, blended


# ─────────────────────────────────────────────────────────────────────────────
# V3: Joint temporal + spatial
# ─────────────────────────────────────────────────────────────────────────────

def mcfm_v3(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool all frames' tokens into one big bank: ((2k+1)*261, 1024).
    frame_t's 261 tokens query over entire bank — one softmax over time+space.
    Output: (261, 1024)
    """
    all_tokens = _stack_window(tokens, window_indices)          # (2k+1, 261, 1024)
    pool       = all_tokens.view(-1, DINO_DIM)                  # ((2k+1)*261, 1024)
    query_t    = _get_tokens(tokens, frame_idx_t)               # (261, 1024)

    scores  = torch.mm(query_t, pool.T) * _SCALE               # (261, (2k+1)*261)
    attn    = torch.softmax(scores, dim=-1)                     # (261, (2k+1)*261)
    blended = torch.mm(attn, pool)                              # (261, 1024)
    return blended, blended


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Temporal THEN Spatial — the mode that was missing
# ─────────────────────────────────────────────────────────────────────────────

def mcfm_temporal_then_spatial(
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1: per-position temporal blend  (identical to mcfm_v2).
    Stage 2: EXPLICIT spatial attention of frame_t against the blended K,V.

    WHY THIS EXISTS. Until now the set was asymmetric: mcfm_v2b did
    spatial->temporal with two explicit stages, but its counterpart mcfm_v2 did
    temporal ONLY, with no spatial stage at all -- spatial mixing was left to the
    frozen model's own cross-attention. Comparing those two answers "one stage or
    two", NOT "which order", which is what the phase7/phase8 write-ups claimed.

    This is the true mirror of mcfm_v2b, and it matches what phase8's
    slat_temporal_then_spatial_v2 does at the SLaT level. With it the 2x2 is
    complete and the ordering question is finally testable at the token level.
    """
    all_tokens = _stack_window(tokens, window_indices)      # (W, N, 1024)
    query_t    = _get_tokens(tokens, frame_idx_t)           # (N, 1024)

    # Stage 1: temporal, per position (same math as mcfm_v2)
    Q  = query_t.unsqueeze(1)                               # (N, 1, D)
    K  = all_tokens.permute(1, 0, 2)                        # (N, W, D)
    at = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * _SCALE, dim=-1)
    kv = torch.bmm(at, K).squeeze(1)                        # (N, D) blended K,V

    # Stage 2: spatial, frame_t queries the temporally-blended bank
    sc = torch.mm(query_t, kv.T) * _SCALE                   # (N, N)
    sp = torch.softmax(sc, dim=-1)
    blended = torch.mm(sp, kv)                              # (N, D)
    return blended, blended


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
# Readable names are the ones to use. The short codes stay as aliases because
# blending_no_lora_no_enhancement/run.py, enhancement/step06_*, step07d_*,
# step10_* and phase_b_baseline/run.py all import mcfm_v1/v2/v2b/v3 BY NAME, and
# every result already on disk is labelled with them. Renaming outright would
# break those call sites and orphan the results.
#
#   temporal_only          token j over token j across the window; NO spatial stage
#                          (spatial mixing left to the model's own cross-attention)
#   temporal_then_spatial  temporal blend, then explicit spatial attention
#   spatial_then_temporal  explicit spatial self-attention, then temporal blend
#   joint_spatiotemporal   one softmax over all W*N tokens: time and space together
#   avg                    fixed weighted sum, no attention (baseline)

mcfm_temporal_only        = mcfm_v2
mcfm_spatial_then_temporal = mcfm_v2b
mcfm_joint_spatiotemporal = mcfm_v3
mcfm_avg                  = mcfm_v1

VERSIONS = {
    # readable
    'avg'                   : mcfm_v1,
    'temporal_only'         : mcfm_v2,
    'temporal_then_spatial' : mcfm_temporal_then_spatial,
    'spatial_then_temporal' : mcfm_v2b,
    'joint_spatiotemporal'  : mcfm_v3,
    # legacy aliases — do not remove, they are on disk in every result label
    'v1' : mcfm_v1,
    'v2' : mcfm_v2,
    'v2b': mcfm_v2b,
    'v3' : mcfm_v3,
}


def run_mcfm(
    version: str,
    tokens: dict,
    window_indices: list,
    frame_idx_t: int,
    lambda_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MCFM version and return (K_hat, V_hat), each (261, 1024)."""
    assert version in VERSIONS, f"Unknown version '{version}'. Choose from {list(VERSIONS)}"
    return VERSIONS[version](tokens, window_indices, frame_idx_t, lambda_vec)
