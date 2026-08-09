"""
Step 6.5 — LoRA adapters for G_L cross-attention (Q, K, V projections).

Rank-4 adapters inserted on every cross_attn block in SLatFlowModel.
All TRELLIS weights are frozen; only LoRA params + alpha are trainable.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Rank-r adapter wrapping a frozen nn.Linear: out = W(x) + B @ A @ x * scale."""
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear  = linear                              # frozen original
        in_f         = linear.in_features
        out_f        = linear.out_features
        self.lora_A  = nn.Parameter(torch.randn(rank, in_f) * 0.02)
        self.lora_B  = nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        return self.linear(x) + (x @ A.T) @ B.T * self.scaling


class LoRABlock(nn.Module):
    """LoRA adapters for one cross_attn block: one for to_q, one for to_kv."""
    def __init__(self, to_q: nn.Linear, to_kv: nn.Linear, rank: int = 4):
        super().__init__()
        self.lora_q  = LoRALinear(to_q,  rank=rank)
        self.lora_kv = LoRALinear(to_kv, rank=rank)


def insert_lora(flow_model, rank: int = 4) -> tuple:
    """
    Freezes all TRELLIS weights and creates LoRABlock for each cross_attn block.

    Returns:
      lora_blocks : nn.ModuleList[LoRABlock]  — one per cross_attn block
      alpha       : nn.Parameter scalar, initialized 0.5
                    (controls temporal path strength in dual-path attention)
    """
    for p in flow_model.parameters():
        p.requires_grad_(False)

    lora_blocks = nn.ModuleList()
    for block in flow_model.blocks:
        if hasattr(block, 'cross_attn'):
            ca = block.cross_attn
            lora_blocks.append(LoRABlock(ca.to_q, ca.to_kv, rank=rank))

    alpha = nn.Parameter(torch.tensor(0.5))
    return lora_blocks, alpha


def trainable_params(lora_blocks: nn.ModuleList, alpha: nn.Parameter) -> list:
    """LoRA A/B matrices + alpha only. Excludes frozen linear weights stored inside LoRALinear."""
    return [p for p in lora_blocks.parameters() if p.requires_grad] + [alpha]


def count_trainable(lora_blocks: nn.ModuleList, alpha: nn.Parameter) -> int:
    return sum(p.numel() for p in trainable_params(lora_blocks, alpha))
