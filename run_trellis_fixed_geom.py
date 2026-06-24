"""
Fixed-geometry TRELLIS texturing for 150 frames.

Step 1: Run sparse structure (geometry) ONCE on the canonical frame.
Step 2: For each frame, run only sample_slat (texture) with fixed geometry.

This eliminates geometry flickering — only texture varies per frame.

Usage:
    python run_trellis_fixed_geom.py \
        --frames_dir  <dir with frame_0001.png ... frame_0150.png> \
        --output_dir  <output root> \
        --canonical_frame 75
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import sys
import argparse
import torch
import numpy as np
import imageio
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--canonical_frame", type=int, default=75,
                        help="Frame index used to generate fixed geometry (1-based)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=150)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TRELLIS pipeline …")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    # ── Step 1: Fix geometry from canonical frame ─────────────────────────────
    canonical_path = frames_dir / f"frame_{args.canonical_frame:04d}.png"
    assert canonical_path.exists(), f"Canonical frame not found: {canonical_path}"

    print(f"Generating fixed geometry from {canonical_path.name} …")
    torch.manual_seed(args.seed)

    canonical_image = Image.open(canonical_path)
    cond_canonical = pipeline.get_cond([canonical_image])

    coords = pipeline.sample_sparse_structure(
        cond_canonical,
        num_samples=1,
        sampler_params={"steps": 12, "cfg_strength": 7.5},
    )
    print(f"  Fixed geometry: {coords.shape[0]} voxels")

    # Save coords so we don't have to recompute if restarting
    torch.save(coords, output_dir / "fixed_coords.pt")

    # ── Step 2: Per-frame texture sampling ────────────────────────────────────
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    frame_paths = [p for p in frame_paths
                   if args.start <= int(p.stem.split("_")[1]) <= args.end]

    print(f"Texturing {len(frame_paths)} frames with fixed geometry …")

    for frame_path in tqdm(frame_paths):
        frame_name = frame_path.stem           # frame_XXXX
        out_dir    = output_dir / frame_name
        glb_path   = out_dir / f"{frame_name}.glb"

        if glb_path.exists():
            tqdm.write(f"  [skip] {frame_name}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(frame_path)
        cond  = pipeline.get_cond([image])

        try:
            # Texture only — geometry (coords) is fixed
            slat = pipeline.sample_slat(
                cond,
                coords,
                sampler_params={"steps": 12, "cfg_strength": 3},
            )

            with torch.no_grad():
                outputs = pipeline.decode_slat(slat, formats=["gaussian", "mesh"])

            # Save GLB
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(str(glb_path))

            tqdm.write(f"  {frame_name} done")

        except Exception:
            import traceback
            tqdm.write(f"  [ERROR] {frame_name}")
            traceback.print_exc()

    print("Done. All GLBs saved.")


if __name__ == "__main__":
    main()
