"""
Latent-space interpolation comparison.

Pipeline:
  1. Load fixed sparse-structure coords (from fixed_geom run).
  2. For each of 150 frames, get DINOv2 cond → run sample_slat with fixed coords
     → save slat feats [N, 8].
  3. Interpolate linearly between slat_frame_0001 and slat_frame_0150.
  4. Compute per-frame L2 distance:  ||slat_interp(i) - slat_gt(i)||
  5. Print + save results to latent_comparison.npz

Usage:
    python compare_latent_interp.py \
        --frames_dir <dir with frame_0001.png ... frame_0150.png> \
        --fixed_coords /path/to/fixed_coords.pt \
        --output_dir   <where to save latents + results>
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import trellis.modules.sparse as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",    required=True)
    parser.add_argument("--fixed_coords",  required=True,
                        help="Path to fixed_coords.pt from the fixed_geom run")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latents_dir = output_dir / "latents"
    latents_dir.mkdir(exist_ok=True)

    print("Loading TRELLIS pipeline …")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    print("Loading fixed coords …")
    coords = torch.load(args.fixed_coords).cuda()
    print(f"  Fixed coords: {coords.shape[0]} voxels")

    # ── Step 1: Encode all 150 frames with fixed coords ───────────────────────
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    print(f"Encoding {len(frame_paths)} frames …")

    torch.manual_seed(args.seed)

    for frame_path in tqdm(frame_paths):
        frame_name = frame_path.stem
        out_path   = latents_dir / f"{frame_name}.npy"

        if out_path.exists():
            continue

        image = Image.open(frame_path)
        cond  = pipeline.get_cond([image])

        with torch.no_grad():
            slat = pipeline.sample_slat(
                cond,
                coords,
                sampler_params={"steps": 12, "cfg_strength": 3},
            )

        np.save(str(out_path), slat.feats.cpu().numpy().astype(np.float32))

    # ── Step 2: Load all latents ──────────────────────────────────────────────
    print("Loading latents …")
    latent_files = sorted(latents_dir.glob("frame_*.npy"))
    latents = np.stack([np.load(str(f)) for f in latent_files])   # [150, N, 8]
    print(f"  Latent array: {latents.shape}")

    # ── Step 3: Linear interpolation between frame_0001 and frame_0150 ───────
    z_start = latents[0]    # frame_0001  [N, 8]
    z_end   = latents[-1]   # frame_0150  [N, 8]

    t_values = np.linspace(0, 1, len(latents))
    z_interp = np.stack([(1 - t) * z_start + t * z_end for t in t_values])  # [150, N, 8]

    # ── Step 4: Per-frame L2 distance ─────────────────────────────────────────
    diff     = z_interp - latents                          # [150, N, 8]
    l2_per_frame  = np.sqrt((diff ** 2).sum(axis=(1, 2))) # [150]
    l2_normalised = l2_per_frame / np.sqrt((latents ** 2).sum(axis=(1, 2)) + 1e-8)

    print("\n── Latent L2 distance: interp vs ground truth ──")
    print(f"  mean  : {l2_per_frame.mean():.4f}")
    print(f"  min   : {l2_per_frame.min():.4f}  (frame {l2_per_frame.argmin()+1:03d})")
    print(f"  max   : {l2_per_frame.max():.4f}  (frame {l2_per_frame.argmax()+1:03d})")
    print(f"  mean normalised: {l2_normalised.mean():.4f}")

    print("\n  Per-frame distances:")
    for i, (d, dn) in enumerate(zip(l2_per_frame, l2_normalised)):
        print(f"    frame_{i+1:04d}  L2={d:.4f}  rel={dn:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        output_dir / "latent_comparison.npz",
        l2_per_frame=l2_per_frame,
        l2_normalised=l2_normalised,
        t_values=t_values,
    )

    # ── Step 5: Plot loss curve ───────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(l2_per_frame) + 1), l2_per_frame, color="crimson", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("L2 distance (interp vs GT)")
    ax.set_title("Latent interpolation error: frame_0001 ↔ frame_0150")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {output_dir}/loss_curve.png")

    # ── Step 6: Decode interpolated latents → GLBs → convert → kaolin render ─
    print("\nDecoding interpolated latents to GLBs …")
    glb_dir = output_dir / "interp_glbs"
    glb_dir.mkdir(exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    for i, t in enumerate(tqdm(t_values)):
        frame_name = f"frame_{i+1:04d}"
        frame_glb_dir = glb_dir / frame_name
        glb_path = frame_glb_dir / f"{frame_name}.glb"
        if glb_path.exists():
            continue
        frame_glb_dir.mkdir(exist_ok=True)

        feats_interp = torch.from_numpy(z_interp[i]).float().cuda()
        slat_interp  = sp.SparseTensor(feats=feats_interp, coords=coords)

        with torch.no_grad():
            outputs = pipeline.decode_slat(slat_interp, formats=["gaussian", "mesh"])

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0],
            simplify=0.95, texture_size=1024,
        )
        glb.export(str(glb_path))

    # Convert GLBs → OBJ
    print("Converting GLBs to OBJ …")
    os.system(
        f"/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python "
        f"/net/projects/ranalab/rajhansini/dynamic_texture/src/convert_glbs_to_obj.py "
        f"--frames_dir {glb_dir}"
    )

    # Render with kaolin (matches original orientation exactly)
    print("Rendering with kaolin …")
    os.system(
        f"MKL_THREADING_LAYER=GNU "
        f"/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python "
        f"/net/projects/ranalab/rajhansini/dynamic_texture/src/render_mvadaptor_render_textured_glb.py "
        f"--frames_dir {glb_dir} --global_normalize"
    )

    # ── Step 7: Stitch side-by-side comparison video ──────────────────────────
    print("Stitching side-by-side comparison video …")
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    for i, frame_path in enumerate(tqdm(frame_paths)):
        out_path = renders_dir / f"frame_{i+1:04d}.png"
        if out_path.exists():
            continue

        frame_name  = f"frame_{i+1:04d}"
        kaolin_render = glb_dir / frame_name / "renders" / "front.png"

        orig_img   = Image.open(frame_path).convert("RGB").resize((512, 512))
        interp_img = Image.open(kaolin_render).convert("RGB").resize((512, 512)) \
                     if kaolin_render.exists() else Image.new("RGB", (512, 512))

        combined = Image.new("RGB", (1024, 512))
        combined.paste(orig_img,   (0,   0))
        combined.paste(interp_img, (512, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((10, 10),  "Original",       fill=(0,   0,   0))
        draw.text((522, 10), "TRELLIS Interp", fill=(0,   0,   0))
        draw.text((10, 490), f"frame {i+1:04d}  L2={l2_per_frame[i]:.3f}", fill=(200, 0, 0))

        combined.save(str(out_path))

    os.system(
        f"ffmpeg -y -framerate 30 "
        f"-pattern_type glob -i '{renders_dir}/frame_*.png' "
        f"-c:v mpeg4 -q:v 5 "
        f"{output_dir}/comparison.mp4"
    )
    print(f"\nDone.")
    print(f"  Video:      {output_dir}/comparison.mp4")
    print(f"  Loss curve: {output_dir}/loss_curve.png")
    print(f"  Results:    {output_dir}/latent_comparison.npz")


if __name__ == "__main__":
    main()
