"""
Creates a minimal fake VideoConditionedSLat dataset so that --tryrun works
without needing real encoded sequences.

Output at: data/dynamic_sequences/seq_dummy/
  metadata.json
  cond.png
  frame_0001/latent.npz
  frame_0002/latent.npz

Usage:
    python scripts/create_dummy_sequences.py [--root data/dynamic_sequences]
"""

import argparse
import json
import os
import numpy as np
from PIL import Image

GRID_SIZE   = 64
N_VOXELS    = 1024   # small but non-trivial
LATENT_DIM  = 8
IMAGE_SIZE  = 518
N_FRAMES    = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dynamic_sequences",
                        help="Root directory for VideoConditionedSLat")
    parser.add_argument("--seq_id", default="seq_dummy")
    args = parser.parse_args()

    seq_dir = os.path.join(args.root, args.seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    print(f"Creating dummy sequence at: {seq_dir}")

    # conditioning image (white 518x518 PNG)
    cond_img = Image.fromarray(
        np.ones((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8) * 255
    )
    cond_path = os.path.join(seq_dir, "cond.png")
    cond_img.save(cond_path)
    print(f"  Wrote {cond_path}")

    frame_names = [f"frame_{i:04d}" for i in range(1, N_FRAMES + 1)]
    tau_values  = [i / max(N_FRAMES - 1, 1) for i in range(N_FRAMES)]

    for frame_name in frame_names:
        frame_dir = os.path.join(seq_dir, frame_name)
        os.makedirs(frame_dir, exist_ok=True)
        latent_path = os.path.join(frame_dir, "latent.npz")

        coords = np.random.randint(0, GRID_SIZE, (N_VOXELS, 3), dtype=np.uint8)
        feats  = np.random.randn(N_VOXELS, LATENT_DIM).astype(np.float32)
        np.savez_compressed(latent_path, coords=coords, feats=feats)
        print(f"  Wrote {latent_path}  ({N_VOXELS} voxels)")

    metadata = {
        "num_frames":  N_FRAMES,
        "tau_values":  tau_values,
        "cond_frame":  1,
        "frame_names": frame_names,
        "global_norm": {"ref_center": [0.0, 0.0, 0.0], "global_scale": 1.0,
                        "dy": 0.25, "target_scale": 0.6},
    }
    meta_path = os.path.join(seq_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {meta_path}")

    print(f"\nDummy sequence ready — {N_FRAMES} frames, "
          f"tau ∈ [{tau_values[0]:.2f}, {tau_values[-1]:.2f}]")


if __name__ == "__main__":
    main()
