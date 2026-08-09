"""
vis_mask.py
-----------
Shared helper: turn the raw gradient scores written by compute_visibility_mask.py
into a per-voxel weight vector w in [0, 1] that multiplies the LoRA colour delta.

    delta_effective[v] = w[v] * delta_lora[v]

w = 1 → voxel was well seen by the front camera, its learned delta is trustworthy
w = 0 → voxel was never seen, its delta is noise, fall back to frozen TRELLIS

Modes
  none  : w = 1 everywhere  (reproduces v4 exactly — the control)
  hard  : w = 1 where score > eps * max(score), else 0
  soft  : w = clamp(score / q, 0, 1) ** gamma, q = a high quantile of the
          nonzero scores.  Voxels at grazing angles get a small score, so they
          fade out instead of stopping at a hard silhouette seam.

Aggregation across the 150 probed frames
  any      : union — w built from max over frames  (default for hard)
  mean     : average over frames                   (default for soft)
  per-frame: use that frame's own scores

'any'/'mean' are recommended over 'per-frame': geometry shifts slightly frame to
frame, so a per-frame visible set makes the lava boundary crawl over time.
"""

import json
from pathlib import Path

import numpy as np
import torch


class VisibilityMask:
    def __init__(self, npz_path):
        npz          = np.load(Path(npz_path), allow_pickle=True)
        # asarray, not astype: scores are already float32, so this avoids a
        # second ~300 MB copy on the training node.
        self.scores  = np.asarray(npz['scores'], dtype=np.float32)   # (F, N_fine)
        self.frames  = npz['frames'].astype(np.int64)     # (F,)
        self.meta    = json.loads(str(npz['meta']))
        self.n_fine  = self.scores.shape[1]
        self._index  = {int(f): i for i, f in enumerate(self.frames)}

        self._agg_any  = self.scores.max(axis=0)
        self._agg_mean = self.scores.mean(axis=0)

    # ── raw score selection ──────────────────────────────────────────────────
    def _raw(self, agg, frame=None):
        if agg == 'any':
            return self._agg_any
        if agg == 'mean':
            return self._agg_mean
        if agg == 'per-frame':
            if frame is None:
                raise ValueError("agg='per-frame' requires a frame index")
            if frame not in self._index:
                # frame was skipped by --stride; fall back to nearest probed frame
                nearest = min(self._index, key=lambda f: abs(f - frame))
                return self.scores[self._index[nearest]]
            return self.scores[self._index[frame]]
        raise ValueError(f'unknown agg: {agg}')

    # ── weights ──────────────────────────────────────────────────────────────
    def weights(self, mode='soft', agg=None, frame=None,
                eps=1e-4, soft_q=0.50, gamma=1.0):
        """Returns float32 numpy (N_fine,) in [0, 1]."""
        if mode == 'none':
            return np.ones(self.n_fine, dtype=np.float32)

        if agg is None:
            agg = 'any' if mode == 'hard' else 'mean'
        s = self._raw(agg, frame)

        if mode == 'hard':
            thr = eps * float(s.max())
            return (s > thr).astype(np.float32)

        if mode == 'soft':
            nz = s[s > 0]
            if nz.size == 0:
                return np.zeros(self.n_fine, dtype=np.float32)
            q = float(np.quantile(nz, soft_q))
            if q <= 0:
                return (s > 0).astype(np.float32)
            w = np.clip(s / q, 0.0, 1.0).astype(np.float32)
            if gamma != 1.0:
                w = w ** gamma
            return w

        raise ValueError(f'unknown mode: {mode}')

    def weights_torch(self, device, **kw):
        return torch.from_numpy(self.weights(**kw)).to(device)

    # ── reporting ────────────────────────────────────────────────────────────
    def describe(self, mode='soft', agg=None, frame=None, **kw):
        w = self.weights(mode=mode, agg=agg, frame=frame, **kw)
        if agg is None:
            agg = 'any' if mode == 'hard' else 'mean'
        return (f'mask[mode={mode} agg={agg}]  '
                f'nonzero={float((w > 0).mean())*100:.2f}%  '
                f'mean_w={float(w.mean()):.4f}  '
                f'w==1={float((w >= 0.999).mean())*100:.2f}%  '
                f'N={w.shape[0]:,}')


def channel_mask(mode='all', color_dim=48, per_corner=6, n_rgb=3):
    """
    Per-channel weight over the 48 "colour" channels at out_layer[53:101].

    Those 48 are NOT all albedo. cube2mesh.SparseFeatures2Mesh lays them out as
    8 cube corners x 6, and the per-corner 6 are 3 albedo + 3 SHADING NORMAL
    ("verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal").
    So flat index i is albedo iff i % 6 < 3.

    Vertex POSITIONS come only from the deform block [8:32], so the LoRA never
    moves geometry either way — but a delta spanning all 48 does perturb the
    shading normals, which is why "colour-only" is imprecise for v4.

      all : all 48        (reproduces v4 exactly — the default)
      rgb : albedo only, shading normals left frozen
    """
    idx = torch.arange(color_dim)
    if mode == 'all':
        return torch.ones(color_dim)
    if mode == 'rgb':
        return ((idx % per_corner) < n_rgb).float()
    raise ValueError(f'unknown channel mode: {mode}')


def add_mask_args(ap):
    """Attach the standard mask CLI flags to an ArgumentParser."""
    ap.add_argument('--mask-npz',  default=None,
                    help='path to visibility.npz (default: <visibility>/masks/visibility.npz)')
    ap.add_argument('--mask-mode', default='soft', choices=['none', 'hard', 'soft'])
    ap.add_argument('--mask-agg',  default=None,
                    choices=['any', 'mean', 'per-frame'])
    ap.add_argument('--mask-eps',  type=float, default=1e-4,
                    help='hard mode: threshold as a fraction of max score')
    ap.add_argument('--mask-q',    type=float, default=0.50,
                    help='soft mode: quantile of nonzero scores mapped to w=1. '
                         '0.5 = the better-seen half of visible voxels keeps the '
                         'delta at full strength; only the low-gradient tail fades.')
    ap.add_argument('--mask-gamma', type=float, default=1.0,
                    help='soft mode: exponent on the ramp (>1 sharpens)')
    ap.add_argument('--delta-channels', default='all', choices=['all', 'rgb'],
                    help="which of the 48 out_layer[53:101] channels the LoRA "
                         "delta may touch. 'all' = v4 behaviour (albedo AND "
                         "shading normals); 'rgb' = albedo only.")
    return ap


def default_mask_npz():
    return Path(__file__).resolve().parent / 'masks' / 'visibility.npz'


def load_mask(args):
    """Build a VisibilityMask from parsed args, or None when mode == 'none'."""
    if args.mask_mode == 'none':
        return None
    path = Path(args.mask_npz) if args.mask_npz else default_mask_npz()
    if not path.exists():
        raise FileNotFoundError(
            f'visibility mask not found: {path}\n'
            f'run compute_visibility_mask.py first.'
        )
    return VisibilityMask(path)


def mask_kwargs(args):
    return dict(mode=args.mask_mode, agg=args.mask_agg,
                eps=args.mask_eps, soft_q=args.mask_q, gamma=args.mask_gamma)
