"""
LoRA adapters — fixed version (Hu et al. standard).

Fixes vs lora_v2.py:
  FIX 1: A init changed from N(0, 1/rank) [||A||~16] to kaiming_uniform [||A||~1.6]
  FIX 2: added (lora_alpha / rank) scaling on output; lora_alpha=rank → scaling=1.0

BACKWARD COMPATIBLE: lora_v2.py untouched. New results dirs used by step07d.
"""
import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Pure delta adapter: out = ((x @ A^T) @ B^T) * scaling
    A: kaiming_uniform (per Hu et al.)
    B: zeros at init → output exactly zero until training moves B
    scaling = lora_alpha / rank = 1.0 (lora_alpha=rank, standard)
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, lora_alpha: int = None):
        super().__init__()
        lora_alpha   = rank if lora_alpha is None else lora_alpha
        self.scaling = lora_alpha / rank                              # 1.0 when lora_alpha=rank
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))             # per Hu et al., ||A||~1.6
        self.B = nn.Parameter(torch.zeros(out_dim, rank))             # zeros — init gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 for low-rank matmuls: prevents grad underflow in B without touching PATH A
        return (((x.float() @ self.A.T) @ self.B.T) * self.scaling).to(x.dtype)


class LoRABlock(nn.Module):
    def __init__(self, dim: int = 1024, rank: int = 4, lora_alpha: int = None):
        super().__init__()
        self.lora_q  = LoRALayer(dim, dim,     rank, lora_alpha)
        self.lora_kv = LoRALayer(dim, 2 * dim, rank, lora_alpha)


def build_lora_blocks(rank: int = 4, n_blocks: int = 24, lora_alpha: int = None) -> nn.ModuleList:
    return nn.ModuleList([LoRABlock(dim=1024, rank=rank, lora_alpha=lora_alpha)
                          for _ in range(n_blocks)])


def freeze_trellis(flow_model) -> None:
    for p in flow_model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in flow_model.parameters())
    print(f'[LORA_V2_FIXED] Froze {n:,} TRELLIS params')


def gate0_verify(lora_blocks: nn.ModuleList) -> None:
    print('\n[GATE 0] Construction verification (lora_v2_fixed)...')
    b0 = lora_blocks[0]
    # B must be zero
    for i, blk in enumerate(lora_blocks):
        assert blk.lora_q.B.abs().max().item()  == 0.0, f'blk{i} lora_q.B not zero'
        assert blk.lora_kv.B.abs().max().item() == 0.0, f'blk{i} lora_kv.B not zero'
    print(f'  B matrices zero : 48/48 ✓')
    # A must be kaiming — check norm is in reasonable range
    a_norm = b0.lora_q.A.norm().item()
    print(f'  lora_q.A norm   : {a_norm:.4f}  (expected ~1.6 for kaiming_uniform rank=4, in=1024)')
    assert a_norm < 5.0, f'A norm={a_norm:.4f} too large — init wrong'
    print(f'  scaling         : {b0.lora_q.scaling:.4f}  (expected 1.0 when lora_alpha=rank)')
    print(f'  shapes: qA={tuple(b0.lora_q.A.shape)} qB={tuple(b0.lora_q.B.shape)} '
          f'kvA={tuple(b0.lora_kv.A.shape)} kvB={tuple(b0.lora_kv.B.shape)}')
    print('[GATE 0] PASSED\n')


def trainable_params(lora_blocks: nn.ModuleList) -> list:
    return [p for p in lora_blocks.parameters() if p.requires_grad]


def count_trainable(lora_blocks: nn.ModuleList) -> int:
    return sum(p.numel() for p in trainable_params(lora_blocks))
