"""
encode_dynamic_sequence.py

Encodes a directory of MVAdaptor-textured GLB frames into per-frame SLAT latents
for training DynamicTextureFlowMatchingTrainer.

Pipeline per frame:
  1. Load mesh (OBJ), apply trainer2 global normalization, map to TRELLIS [-0.5,0.5]³
  2. Voxelize with open3d at 64³ resolution
  3. Project voxels back to trainer2 space, cast into each of 6 renders
  4. Extract DINOv2 patch tokens, average across visible views
  5. Run SLAT encoder → latent.npz

Output directory layout:
  <output_dir>/
    metadata.json          ← {num_frames, tau_values, cond_frame}
    cond.png               ← front.png of frame_0001 (reference image)
    frame_0001/latent.npz  ← {coords: (N,3) uint8, feats: (N,8) float32}
    frame_0002/latent.npz
    ...

Usage:
    python dataset_toolkits/encode_dynamic_sequence.py \
        --frames_dir /path/to/trellis_glbs \
        --output_dir /path/to/dynamic_sequences/sequence_001 \
        [--enc_pretrained microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16] \
        [--radius 1.5] [--cond_frame 1]
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import trellis.models as models
import trellis.modules.sparse as sp

torch.set_grad_enabled(False)

# ── Camera parameters (verbatim from render_mvadaptor_render_textured_glb.py) ─
VIEW_NAMES   = ["front", "front_right", "right", "back", "left", "front_left"]
VIEW_AZIMUTHS = [90.0,    45.0,          0.0,     -90.0,  -180.0, -225.0]
ELEVATION_DEG = 0.0
RENDER_SIZE   = 1024       # render script default
FOV_RAD       = np.pi / 3  # kaolin generate_perspective_projection(np.pi/3)

# ── Trainer2 normalization params (from render script) ────────────────────────
DY           = 0.25
TARGET_SCALE = 0.6         # half-extent of the normalized mesh

# ── TRELLIS voxelization params ───────────────────────────────────────────────
GRID_SIZE    = 64
MAX_VOXELS   = 32768

# ── DINOv2 feature extraction ─────────────────────────────────────────────────
DINOV2_MODEL = 'dinov2_vitl14_reg'
DINOV2_SIZE  = 518
N_PATCH      = DINOV2_SIZE // 14   # 37
DINOV2_NORM  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# ─────────────────────────────────────────────────────────────────────────────
# Camera math
# ─────────────────────────────────────────────────────────────────────────────

def _world_to_camera_opencv(azim_deg: float, elev_deg: float, radius: float) -> np.ndarray:
    """
    4x4 world-to-camera in OpenCV convention (X right, Y down, Z forward).
    Matches trainer2/render.py's Y-up spherical camera:
        x = r*cos(e)*cos(a),  y = r*sin(e),  z = r*cos(e)*sin(a)
    """
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    pos  = np.array([
        radius * np.cos(elev) * np.cos(azim),
        radius * np.sin(elev),
        radius * np.cos(elev) * np.sin(azim),
    ], dtype=np.float64)

    # Camera z-axis points FROM target toward camera (OpenGL backward convention)
    z_gl = pos / np.linalg.norm(pos)
    up   = np.array([0.0, 1.0, 0.0])
    x_gl = np.cross(up, z_gl);  x_gl /= np.linalg.norm(x_gl)
    y_gl = np.cross(z_gl, x_gl)

    R_gl = np.stack([x_gl, y_gl, z_gl], axis=0)   # rows = camera axes (OpenGL)
    t_gl = -R_gl @ pos

    w2c_gl = np.eye(4)
    w2c_gl[:3, :3] = R_gl
    w2c_gl[:3, 3]  = t_gl

    # OpenGL → OpenCV: negate Y and Z rows
    flip          = np.diag([1.0, -1.0, -1.0, 1.0])
    return (flip @ w2c_gl).astype(np.float32)


def _intrinsics(fov_rad: float, image_size: int) -> np.ndarray:
    """3x3 pinhole K for a square image."""
    f = image_size / (2.0 * np.tan(fov_rad / 2.0))
    c = image_size / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float32)


def _project(positions: np.ndarray, w2c: np.ndarray, K: np.ndarray,
             image_size: int) -> tuple:
    """
    Project (N,3) world positions to grid_sample UV in [-1,1].
    Returns (uv: (N,2), valid: (N,) bool).
    """
    N    = positions.shape[0]
    pts  = np.concatenate([positions, np.ones((N, 1), np.float32)], axis=1)  # (N,4)
    cam  = (w2c @ pts.T).T[:, :3]   # (N,3), OpenCV: z>0 = in front

    valid = cam[:, 2] > 0.0

    uvz  = (K @ cam.T).T             # (N,3)
    uv_px = uvz[:, :2] / np.maximum(uvz[:, 2:3], 1e-6)   # pixel coords
    uv   = (uv_px / image_size) * 2.0 - 1.0               # [-1,1]

    valid &= ((uv[:, 0] > -1) & (uv[:, 0] < 1) &
              (uv[:, 1] > -1) & (uv[:, 1] < 1))
    return uv.astype(np.float32), valid


# ─────────────────────────────────────────────────────────────────────────────
# Mesh normalization
# ─────────────────────────────────────────────────────────────────────────────

def _compute_global_norm(frame_dirs: list) -> tuple:
    """Lock center to frame_0001, global scale = max norm across all frames."""
    centers, scales = [], []
    for fd in frame_dirs:
        m = o3d.io.read_triangle_mesh(str(fd / "converted" / "mesh.obj"))
        v = np.asarray(m.vertices, np.float32)
        c = v.mean(axis=0)
        centers.append(c)
        scales.append(np.linalg.norm(v - c, axis=1).max())
    return np.array(centers[0], np.float32), float(max(scales))


def _normalize_trainer2(verts: np.ndarray, ref_center: np.ndarray,
                         global_scale: float) -> np.ndarray:
    """Same normalization as the render script."""
    v = (verts - ref_center) / global_scale * TARGET_SCALE
    v[:, 1] += DY
    return v


def _trainer2_to_trellis(verts_t2: np.ndarray) -> np.ndarray:
    """Map trainer2 space → TRELLIS [-0.5, 0.5]³."""
    return (verts_t2 - np.array([0.0, DY, 0.0], np.float32)) / (2.0 * TARGET_SCALE)


def _trellis_to_trainer2(verts_tr: np.ndarray) -> np.ndarray:
    return verts_tr * (2.0 * TARGET_SCALE) + np.array([0.0, DY, 0.0], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Voxelization
# ─────────────────────────────────────────────────────────────────────────────

def _voxelize(mesh_path: Path, ref_center: np.ndarray,
              global_scale: float) -> np.ndarray:
    """
    Returns voxel centers in TRELLIS space (N, 3) ∈ [-0.5, 0.5]³.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts_raw = np.asarray(mesh.vertices, np.float32)
    verts_t2  = _normalize_trainer2(verts_raw, ref_center, global_scale)
    verts_tr  = np.clip(_trainer2_to_trellis(verts_t2), -0.5 + 1e-6, 0.5 - 1e-6)

    # Rebuild mesh with trellis-space vertices (keep faces for solid voxelization)
    mesh_tr = o3d.geometry.TriangleMesh()
    mesh_tr.vertices  = o3d.utility.Vector3dVector(verts_tr.astype(np.float64))
    mesh_tr.triangles = mesh.triangles

    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_tr, voxel_size=1.0 / GRID_SIZE,
        min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5),
    )
    positions = np.array(
        [(v.grid_index + 0.5) / GRID_SIZE - 0.5 for v in vg.get_voxels()],
        dtype=np.float32,
    )
    return positions   # (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(
    render_dir: Path,
    positions_trellis: np.ndarray,
    cameras: list,          # [(w2c_opencv, K)] per view
    dinov2,
    device: torch.device,
) -> np.ndarray:
    """
    Returns averaged per-voxel DINOv2 features (N, 1024) as float16.
    """
    N = positions_trellis.shape[0]
    # Project in trainer2 space (where the renders live)
    positions_t2 = _trellis_to_trainer2(positions_trellis)

    feat_sum = np.zeros((N, 1024), np.float32)
    vis_cnt  = np.zeros(N, np.int32)

    for view_name, (w2c, K) in zip(VIEW_NAMES, cameras):
        img_path = render_dir / f"{view_name}.png"
        if not img_path.exists():
            continue

        # Load render → DINOv2 input
        img = Image.open(img_path).convert("RGB")
        img = img.resize((DINOV2_SIZE, DINOV2_SIZE), Image.Resampling.LANCZOS)
        img_t = transforms.ToTensor()(img)       # (3, 518, 518)
        img_t = DINOV2_NORM(img_t).unsqueeze(0).to(device)

        with torch.no_grad():
            out = dinov2(img_t, is_training=True)
        # patch tokens: (1, num_tokens, 1024) → (1, 1024, n_patch, n_patch)
        tokens = out['x_prenorm'][:, dinov2.num_register_tokens + 1:]  # (1, 37²+reg, 1024)
        patches = tokens.permute(0, 2, 1).reshape(1, 1024, N_PATCH, N_PATCH)  # (1,1024,37,37)

        # Project voxels using RENDER_SIZE camera but normalize to [-1,1]
        # (equivalent regardless of which image resolution we use for K)
        uv, valid = _project(positions_t2, w2c, K, RENDER_SIZE)

        # Sample patch features
        uv_t = torch.from_numpy(uv).to(device).view(1, N, 1, 2)   # (1, N, 1, 2)
        sampled = F.grid_sample(
            patches.float(), uv_t, mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(-1).T.cpu().numpy()  # (N, 1024)

        sampled[~valid] = 0.0
        feat_sum += sampled
        vis_cnt  += valid.astype(np.int32)

    vis_safe = np.maximum(vis_cnt, 1)[:, np.newaxis]
    return (feat_sum / vis_safe).astype(np.float16)   # (N, 1024) float16


# ─────────────────────────────────────────────────────────────────────────────
# SLAT encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_latent(
    positions_trellis: np.ndarray,   # (N, 3) float32
    features: np.ndarray,            # (N, 1024) float16
    encoder,
    device: torch.device,
) -> dict:
    """Run SLAT encoder → {coords: (M,3) uint8, feats: (M, latent_dim) float32}."""
    N       = positions_trellis.shape[0]
    indices = np.clip(
        ((positions_trellis + 0.5) * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1
    )  # (N, 3)

    sparse_in = sp.SparseTensor(
        feats=torch.from_numpy(features).float().to(device),
        coords=torch.cat([
            torch.zeros(N, 1, dtype=torch.int32),
            torch.from_numpy(indices).int(),
        ], dim=1).to(device),
    )

    with torch.no_grad():
        latent = encoder(sparse_in, sample_posterior=False)

    assert torch.isfinite(latent.feats).all(), "Non-finite latent — check normalization"
    return {
        'feats':  latent.feats.cpu().numpy().astype(np.float32),
        'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True,
                        help='Directory containing frame_XXXX/ subdirectories '
                             '(each with converted/mesh.obj and renders/*.png)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for the encoded sequence')
    parser.add_argument('--enc_pretrained',
                        default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained SLAT encoder')
    parser.add_argument('--radius', type=float, default=1.5,
                        help='Camera radius used when rendering (default: 1.5)')
    parser.add_argument('--cond_frame', type=int, default=1,
                        help='Which frame to use as the conditioning image (1-indexed)')
    args = parser.parse_args()

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_dir = Path(args.frames_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover frames
    frame_dirs = sorted([
        d for d in frames_dir.iterdir()
        if d.is_dir() and d.name.startswith('frame_')
        and (d / 'converted' / 'mesh.obj').exists()
        and (d / 'renders' / 'front.png').exists()
    ])
    if not frame_dirs:
        raise RuntimeError(f"No valid frame directories found under {frames_dir}")
    print(f"Found {len(frame_dirs)} frames")

    # Global normalization (matches render script)
    print("Computing global normalization...")
    ref_center, global_scale = _compute_global_norm(frame_dirs)
    print(f"  ref_center={ref_center}, global_scale={global_scale:.6f}")

    # Precompute camera matrices (same for all frames)
    K       = _intrinsics(FOV_RAD, RENDER_SIZE)
    cameras = [
        (_world_to_camera_opencv(azim, ELEVATION_DEG, args.radius), K)
        for azim in VIEW_AZIMUTHS
    ]

    # Load models
    print("Loading DINOv2...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL)
    dinov2.eval().to(device)

    print(f"Loading SLAT encoder from {args.enc_pretrained}...")
    encoder = models.from_pretrained(args.enc_pretrained).eval().to(device)

    # Save conditioning image (front view of cond_frame)
    cond_idx  = args.cond_frame - 1
    cond_src  = frame_dirs[cond_idx] / 'renders' / 'front.png'
    cond_dst  = out_dir / 'cond.png'
    shutil.copy(cond_src, cond_dst)
    print(f"Conditioning image: {cond_src} → {cond_dst}")

    # Encode each frame
    tau_values = []
    T = len(frame_dirs)

    for idx, frame_dir in enumerate(tqdm(frame_dirs, desc='Encoding frames')):
        tau = float(idx) / max(T - 1, 1)
        tau_values.append(tau)

        frame_out = out_dir / frame_dir.name
        frame_out.mkdir(exist_ok=True)
        latent_path = frame_out / 'latent.npz'

        if latent_path.exists():
            print(f"  {frame_dir.name}: already encoded, skipping")
            continue

        mesh_path  = frame_dir / 'converted' / 'mesh.obj'
        render_dir = frame_dir / 'renders'

        # 1. Voxelize
        positions = _voxelize(mesh_path, ref_center, global_scale)
        if len(positions) == 0:
            print(f"  {frame_dir.name}: WARNING — voxelization empty, skipping")
            continue
        if len(positions) > MAX_VOXELS:
            # Random subsample to keep memory manageable
            idx_sub = np.random.choice(len(positions), MAX_VOXELS, replace=False)
            positions = positions[idx_sub]

        # 2. Extract DINOv2 features
        features = _extract_features(render_dir, positions, cameras, dinov2, device)

        # 3. Encode to SLAT latent
        latent = _encode_latent(positions, features, encoder, device)

        np.savez_compressed(latent_path, **latent)

    # Save metadata
    metadata = {
        'num_frames':  T,
        'tau_values':  tau_values,
        'cond_frame':  args.cond_frame,
        'frame_names': [d.name for d in frame_dirs],
        'global_norm': {
            'ref_center':   ref_center.tolist(),
            'global_scale': global_scale,
            'dy':           DY,
            'target_scale': TARGET_SCALE,
        },
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Sequence saved to {out_dir}")
    print(f"  {T} frames, tau ∈ [{tau_values[0]:.3f}, {tau_values[-1]:.3f}]")


if __name__ == '__main__':
    main()
