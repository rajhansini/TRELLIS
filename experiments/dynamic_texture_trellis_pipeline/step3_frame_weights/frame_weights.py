"""
Step 3: Generate Frame Weights via lightweight projector.

INPUT:
  t             : float 0.0 -> 1.0
  frame_idx_t   : int, target frame index (from Step 1)
  window_indices: list[int], 2k+1 frame indices (from Step 1)
  tokens        : dict {frame_idx: {"cls", "reg", "patches"}} (from Step 2)
  projector     : FrameWeightProjector (trainable)

OUTPUT:
  lambda_dict : dict {frame_idx: scalar weight}  (2k+1 entries)
  lambda_vec  : tensor (2k+1,) softmax weights

Architecture per k:
  Input (1025) -> Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(2k+1) -> Softmax
"""

import torch
import torch.nn as nn


class FrameWeightProjector(nn.Module):
    """
    Lightweight MLP: (t_scalar + CLS embedding) -> frame weights.

    Input : 1025 = 1 (t) + 1024 (CLS of target frame)
    Output: 2k+1 weights (softmaxed)
    """

    def __init__(self, k: int, hidden1: int = 512, hidden2: int = 256):
        super().__init__()
        self.k       = k
        self.n_frames = 2 * k + 1
        self.net = nn.Sequential(
            nn.Linear(1025, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, self.n_frames),
        )

    def forward(self, t: float, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
          t             : scalar float 0.0 -> 1.0
          cls_embedding : tensor (1024,) — CLS token of target frame

        Returns:
          lambda_vec : tensor (2k+1,) softmax weights, sums to 1
        """
        t_tensor = torch.tensor([t], dtype=torch.float32, device=cls_embedding.device)
        x        = torch.cat([t_tensor, cls_embedding], dim=0)   # (1025,)
        logits   = self.net(x)                                    # (2k+1,)
        return torch.softmax(logits, dim=0)                       # (2k+1,)


def get_frame_weights(
    t: float,
    frame_idx_t: int,
    window_indices: list,
    tokens: dict,
    projector: FrameWeightProjector,
) -> tuple[torch.Tensor, dict]:
    """
    Run projector to get per-frame weights.

    Returns:
      lambda_vec  : tensor (2k+1,) summing to 1
      lambda_dict : dict {frame_idx: scalar tensor}
    """
    # PI's note: start with λ = 1 fixed (uniform weights), no learning.
    # Projector-based lambda kept below for when we want to train it.
    # cls_embedding = tokens[frame_idx_t]['cls'].squeeze(0)   # (1024,)
    # lambda_vec    = projector(t, cls_embedding)              # (2k+1,)
    device     = tokens[frame_idx_t]['cls'].device
    lambda_vec = torch.full((projector.n_frames,), 1.0 / projector.n_frames, device=device)

    # At boundaries, duplicate indices get summed (not overwritten)
    lambda_dict: dict = {}
    for i, idx in enumerate(window_indices):
        if idx in lambda_dict:
            lambda_dict[idx] = lambda_dict[idx] + lambda_vec[i]
        else:
            lambda_dict[idx] = lambda_vec[i]
    return lambda_vec, lambda_dict
