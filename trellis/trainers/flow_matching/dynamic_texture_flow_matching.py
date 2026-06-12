from typing import *

import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from .sparse_flow_matching import ImageConditionedSparseFlowMatchingCFGTrainer


class DynamicTextureFlowMatchingTrainer(ImageConditionedSparseFlowMatchingCFGTrainer):
    """
    Trainer for dynamic texture generation via temporal modulation coefficients.

    Extends image-conditioned sparse flow matching with a temporal scalar τ ∈ [0, 1]
    injected into the denoiser's timestep embedding via TemporalProjector. Trained with
    per-frame latent supervision derived from video sequences.

    Each batch item carries a `tau` value (its normalized position in the video).
    If `tau` is absent from the batch, it is sampled uniformly from [0, 1].

    Additional Args:
        lambda_smooth (float): Weight for temporal smoothness regularisation. Default 0.1.
            Penalises large Δmod jumps between τ and τ + ε, encouraging smooth texture motion.
        smooth_eps (float): Finite-difference step used for the smoothness loss. Default 0.05.
    """

    def __init__(self, *args, lambda_smooth: float = 0.1, smooth_eps: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_smooth = lambda_smooth
        self.smooth_eps = smooth_eps

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        tau: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single step.

        Args:
            x_0: Sparse structured latent at time τ (clean).
            cond: Reference image(s) for image conditioning.
            tau: Temporal positions ∈ [0, 1] for each batch item.  Sampled
                 uniformly if not provided.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        if tau is None:
            tau = torch.rand(x_0.shape[0], device=x_0.device)
        else:
            tau = tau.to(x_0.device).float()

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, tau=tau)
        assert pred.feats.shape == noise.feats.shape == x_0.feats.shape
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # temporal smoothness: consecutive τ values should yield similar Δmod
        projector = self.models['denoiser'].temporal_projector
        if self.lambda_smooth > 0 and projector is not None:
            tau_next = (tau + self.smooth_eps).clamp(0.0, 1.0)
            cond_fp32 = cond.float()
            delta_now = projector(tau, cond_fp32)
            delta_next = projector(tau_next, cond_fp32)
            terms["smooth"] = F.mse_loss(delta_now, delta_next)
            terms["loss"] = terms["loss"] + self.lambda_smooth * terms["smooth"]

        # per-time-bin mse for diagnostics
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
