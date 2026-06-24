"""
Run full TRELLIS pipeline on 150 extracted video frames.
Input:  all_frames_150/frame_XXXX.png
Output: trellis_150_frames/frame_XXXX/frame_XXXX.glb
        trellis_150_frames/frame_XXXX/render_front.png
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import sys
import argparse
import numpy as np
import imageio
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True,
                        help="Dir with frame_XXXX.png (e.g. all_frames_150/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output root (e.g. trellis_150_frames/)")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=150)
    parser.add_argument("--seed",  type=int, default=1)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TRELLIS pipeline …")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    frame_paths = [p for p in frame_paths
                   if args.start <= int(p.stem.split("_")[1]) <= args.end]

    print(f"Processing {len(frame_paths)} frames …")

    for frame_path in tqdm(frame_paths):
        frame_name = frame_path.stem          # frame_XXXX
        out_dir = output_dir / frame_name
        glb_path = out_dir / f"{frame_name}.glb"

        if glb_path.exists():
            tqdm.write(f"  [skip] {frame_name}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(frame_path)

        try:
            outputs = pipeline.run(
                image,
                seed=args.seed,
                sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                slat_sampler_params={"steps": 12, "cfg_strength": 3},
            )

            # Save GLB
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(str(glb_path))

            # Save a front-view render for quick comparison
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            # frame 0 of the 360 video ≈ front view
            Image.fromarray(video[0]).save(str(out_dir / "render_front.png"))

            tqdm.write(f"  {frame_name} done")

        except Exception:
            import traceback
            tqdm.write(f"  [ERROR] {frame_name}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
