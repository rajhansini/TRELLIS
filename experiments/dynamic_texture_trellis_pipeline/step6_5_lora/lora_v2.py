"""
Step 6.5 v2 — LoRA adapters (delta-only, clean separation from frozen weights).

BACKWARD COMPATIBLE: original lora.py untouched.

Architecture:
  LoRALayer : pure delta x -> (x @ A^T) @ B^T   (no frozen linear inside)
  LoRABlock : lora_q  (1024 -> 1024,  8,192 params)
            + lora_kv (1024 -> 2048, 12,288 params)
  24 blocks + 1 alpha = 491,521 trainable params  ← GATE 0

Init:
  A ~ Gaussian(0, 1/rank)   (non-zero so grad flows through B when B becomes non-zero)
  B = zeros                  (delta = A^T @ B^T @ x = 0 at init — CRITICAL)
  alpha = 0.5

B=zeros guarantees: at init, PATH B output = 0, combined output = PATH A only.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Pure delta adapter: out = (x @ A^T) @ B^T.
    B=zeros at init → output is exactly zero until training moves B.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) * (1.0 / rank))  # gaussian, non-zero
        self.B = nn.Parameter(torch.zeros(out_dim, rank))                 # ZEROS — init gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        return (x @ A.T) @ B.T    # (N, out_dim), exactly 0 at init


class LoRABlock(nn.Module):
    """One LoRABlock per cross-attn block. Holds Q-adapter and KV-adapter."""
    def __init__(self, dim: int = 1024, rank: int = 4):
        super().__init__()
        self.lora_q  = LoRALayer(dim, dim,     rank)   # 1024→1024 :  8,192 params
        self.lora_kv = LoRALayer(dim, 2 * dim, rank)   # 1024→2048 : 12,288 params


def build_lora_blocks(rank: int = 4, n_blocks: int = 24) -> nn.ModuleList:
    """Build 24 LoRABlocks. Call BEFORE freezing the flow model (freezing is separate)."""
    return nn.ModuleList([LoRABlock(dim=1024, rank=rank) for _ in range(n_blocks)])


def freeze_trellis(flow_model) -> None:
    """Freeze ALL parameters of the TRELLIS flow model."""
    for p in flow_model.parameters():
        p.requires_grad_(False)
    n_frozen = sum(p.numel() for p in flow_model.parameters())
    print(f'[LORA_V2] Froze {n_frozen:,} TRELLIS params (requires_grad=False)')


def gate0_verify(lora_blocks: nn.ModuleList, alpha: nn.Parameter) -> None:
    """
    GATE 0 — construction checks. Call immediately after building lora_blocks.
    Raises AssertionError if anything is wrong.
    """
    print('\n[GATE 0] Construction verification...')

    # 1. Trainable param count
    trainable = sum(p.numel() for p in lora_blocks.parameters() if p.requires_grad) + 1
    assert trainable == 491_521, f'[GATE 0 FAIL] param count={trainable}, expected 491,521'
    print(f'  [GATE 0] trainable params : {trainable}  ✓')

    # 2. All B matrices exactly zero
    for i, blk in enumerate(lora_blocks):
        assert blk.lora_q.B.abs().max().item()  == 0.0, f'[GATE 0 FAIL] block {i} lora_q.B not zero'
        assert blk.lora_kv.B.abs().max().item() == 0.0, f'[GATE 0 FAIL] block {i} lora_kv.B not zero'
    print(f'  [GATE 0] B matrices zero  : 48/48  ✓  (24 lora_q + 24 lora_kv)')

    # 3. alpha init
    print(f'  [GATE 0] alpha init       : {alpha.item():.4f}  (expected 0.5)')

    # 4. No requires_grad on alpha's value (it's a Parameter so it has grad by default — fine)
    print(f'  [GATE 0] alpha.requires_grad : {alpha.requires_grad}  (expected True)')

    # 5. Per-block shapes
    b0 = lora_blocks[0]
    print(f'  [GATE 0] lora_q  A shape : {tuple(b0.lora_q.A.shape)}   (expected (4, 1024))')
    print(f'  [GATE 0] lora_q  B shape : {tuple(b0.lora_q.B.shape)}   (expected (1024, 4))')
    print(f'  [GATE 0] lora_kv A shape : {tuple(b0.lora_kv.A.shape)}  (expected (4, 1024))')
    print(f'  [GATE 0] lora_kv B shape : {tuple(b0.lora_kv.B.shape)}  (expected (2048, 4))')

    print('[GATE 0] PASSED\n')


def trainable_params_v2(lora_blocks: nn.ModuleList, alpha: nn.Parameter) -> list:
    """Return list of trainable parameters for optimizer."""
    return [p for p in lora_blocks.parameters() if p.requires_grad] + [alpha]


def count_trainable_v2(lora_blocks: nn.ModuleList, alpha: nn.Parameter) -> int:
    return sum(p.numel() for p in trainable_params_v2(lora_blocks, alpha))
