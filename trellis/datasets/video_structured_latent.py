"""
Video-conditioned structured latent dataset for dynamic texture training.

Each dataset "instance" is a video sequence encoded by encode_dynamic_sequence.py:

    <sequence_root>/
        <seq_id>/
            metadata.json          ← {num_frames, tau_values, cond_frame, frame_names}
            cond.png               ← reference conditioning image (front view, frame 0)
            frame_0001/latent.npz  ← {coords: (N,3) uint8, feats: (N,C) float32}
            frame_0002/latent.npz
            ...

Each call to __getitem__ returns a randomly sampled (frame_latent, cond_image, tau) triple
from one sequence.
"""

import os
import json
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices


class VideoConditionedSLat(Dataset):
    """
    Dataset of temporally encoded SLAT sequences for dynamic texture training.

    Args:
        roots (str | list[str]): One or more sequence root directories. Each must
            contain subdirectories produced by encode_dynamic_sequence.py.
        image_size (int): Resize conditioning image to this square size before returning.
        normalization (dict): Optional {'mean': [...], 'std': [...]} to normalise latent feats.
        min_frames (int): Skip sequences with fewer than this many encoded frames.
    """

    def __init__(
        self,
        roots: Union[str, List[str]],
        *,
        image_size: int = 518,
        normalization: Optional[dict] = None,
        min_frames: int = 2,
        # kept for config compatibility with other SLat datasets
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.image_size    = image_size
        self.normalization = normalization
        self.min_frames    = min_frames

        if isinstance(roots, str):
            roots = [roots]

        # Discover all valid sequences
        self._sequences: List[dict] = []
        for root in roots:
            root = os.path.expanduser(root)
            for seq_id in sorted(os.listdir(root)):
                seq_path  = os.path.join(root, seq_id)
                meta_path = os.path.join(seq_path, 'metadata.json')
                cond_path = os.path.join(seq_path, 'cond.png')
                if not (os.path.isdir(seq_path) and
                        os.path.exists(meta_path) and
                        os.path.exists(cond_path)):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                valid_frames = [
                    fn for fn in meta['frame_names']
                    if os.path.exists(os.path.join(seq_path, fn, 'latent.npz'))
                ]
                if len(valid_frames) < min_frames:
                    continue
                tau_map = {fn: tau for fn, tau in
                           zip(meta['frame_names'], meta['tau_values'])}
                self._sequences.append({
                    'seq_path':    seq_path,
                    'cond_path':   cond_path,
                    'frame_names': valid_frames,
                    'tau_map':     tau_map,
                })

        if not self._sequences:
            raise ValueError(f"No valid sequences found under {roots}")

        if normalization is not None:
            self.mean = torch.tensor(normalization['mean']).reshape(1, -1)
            self.std  = torch.tensor(normalization['std']).reshape(1, -1)

        # Flat index: (seq_idx, frame_name)
        self._index: List[Tuple[int, str]] = [
            (si, fn)
            for si, seq in enumerate(self._sequences)
            for fn in seq['frame_names']
        ]
        self.loads = self._precompute_loads()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        si, frame_name = self._index[idx]
        seq = self._sequences[si]

        data   = np.load(os.path.join(seq['seq_path'], frame_name, 'latent.npz'))
        coords = torch.from_numpy(data['coords'].astype(np.int32))
        feats  = torch.from_numpy(data['feats'].astype(np.float32))

        if self.normalization is not None:
            feats = (feats - self.mean) / self.std

        cond = self._load_cond(seq['cond_path'])
        tau  = torch.tensor(seq['tau_map'][frame_name], dtype=torch.float32)

        return {'coords': coords, 'feats': feats, 'cond': cond, 'tau': tau}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_cond(self, cond_path: str) -> torch.Tensor:
        """Load and resize the conditioning image to (3, image_size, image_size)."""
        img = Image.open(cond_path).convert('RGBA')

        # White-background composite (same as ImageConditionedMixin)
        img_arr  = np.array(img).astype(np.float32) / 255.0
        alpha    = img_arr[:, :, 3:]
        rgb      = img_arr[:, :, :3]
        img_arr  = rgb * alpha + (1.0 - alpha)   # white background

        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        return transforms.ToTensor()(img)   # (3, H, W) in [0, 1]

    def _precompute_loads(self) -> List[int]:
        """Return approximate voxel counts for load-balanced batching."""
        loads = []
        for si, fn in self._index:
            seq = self._sequences[si]
            latent_path = os.path.join(seq['seq_path'], fn, 'latent.npz')
            # Peek at the file to get voxel count without loading all feats
            try:
                d = np.load(latent_path)
                loads.append(int(d['coords'].shape[0]))
            except Exception:
                loads.append(1)
        return loads

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list, split_size=None) -> Union[dict, List[dict]]:
        if split_size is None:
            groups = [list(range(len(batch)))]
        else:
            loads  = [b['coords'].shape[0] for b in batch]
            groups = load_balanced_group_indices(loads, split_size)

        packs = []
        for group in groups:
            sub = [batch[i] for i in group]
            coords, feats, layout = [], [], []
            start = 0
            for i, b in enumerate(sub):
                coords.append(torch.cat(
                    [torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']],
                    dim=1,
                ))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]

            x_0 = SparseTensor(
                coords=torch.cat(coords),
                feats=torch.cat(feats),
            )
            x_0._shape = torch.Size([len(sub), *sub[0]['feats'].shape[1:]])
            x_0.register_spatial_cache('layout', layout)

            pack = {
                'x_0':  x_0,
                'cond': torch.stack([b['cond'] for b in sub]),   # (B, 3, H, W)
                'tau':  torch.stack([b['tau']  for b in sub]),   # (B,)
            }
            packs.append(pack)

        return packs[0] if split_size is None else packs
