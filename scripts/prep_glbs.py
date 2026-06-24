"""
Converts a flat directory of GLBs into the frame_XXXX/ structure that
encode_dynamic_sequence.py expects.

Input layout (GLBs sorted alphabetically = temporal order):
    glbs_dir/
        something_001.glb
        something_002.glb
        ...

Output layout:
    output_dir/
        frame_0001/
            converted/mesh.obj   (+ mesh.mtl, textures)
            renders/front.png
            renders/front_right.png
            renders/right.png
            renders/back.png
            renders/left.png
            renders/front_left.png
        frame_0002/
        ...

Usage:
    python scripts/prep_glbs.py \\
        --glbs_dir /path/to/glbs \\
        --output_dir /path/to/frames_dir \\
        [--radius 1.5] [--render_size 1024]
"""

import os, sys, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from PIL import Image

# Camera params matching encode_dynamic_sequence.py exactly
VIEW_NAMES    = ["front", "front_right", "right", "back", "left", "front_left"]
VIEW_AZIMUTHS = [90.0,    45.0,          0.0,     -90.0,  -180.0, -225.0]
ELEVATION_DEG = 0.0

# ── Mesh conversion ───────────────────────────────────────────────────────────

def glb_to_obj(glb_path: Path, out_dir: Path):
    """Export GLB → OBJ (+MTL+textures) using trimesh."""
    import trimesh
    scene = trimesh.load(str(glb_path), force='scene')
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "mesh.obj"

    # Merge all meshes into one for simplicity
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values()
                  if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not meshes:
            raise ValueError(f"No valid meshes found in {glb_path}")
        # Apply scene transforms
        dump = []
        for name, geom in scene.geometry.items():
            node_xforms = scene.graph.get(frame_to=name)
            if node_xforms is not None:
                T = node_xforms[0]
                geom = geom.copy()
                geom.apply_transform(T)
            dump.append(geom)
        mesh = trimesh.util.concatenate(dump) if len(dump) > 1 else dump[0]
    else:
        mesh = scene

    # Export
    export_dict = trimesh.exchange.obj.export_obj(mesh, include_texture=True)
    with open(obj_path, 'w') as f:
        f.write(export_dict['obj'])
    if 'mtl' in export_dict and export_dict['mtl']:
        with open(out_dir / "mesh.mtl", 'w') as f:
            f.write(export_dict['mtl'])
    for fname, data in export_dict.get('textures', {}).items():
        with open(out_dir / fname, 'wb') as f:
            f.write(data)

    return obj_path


# ── Rendering ─────────────────────────────────────────────────────────────────

def _normalize_mesh_o3d(mesh_o3d):
    """Center and scale to fit in [-0.5, 0.5]^3."""
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh_o3d.translate(-center)
    extent = bbox.get_extent().max()
    if extent > 0:
        mesh_o3d.scale(1.0 / extent, center=[0, 0, 0])
    return mesh_o3d


def render_views_open3d(obj_path: Path, renders_dir: Path,
                        radius: float, render_size: int):
    """Render 6 fixed views using open3d OffscreenRenderer (needs EGL/GPU)."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh = _normalize_mesh_o3d(mesh)

    renderer = o3d.visualization.rendering.OffscreenRenderer(render_size, render_size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if mesh.has_textures() else "defaultLitSSR"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # White background
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    # Sun light
    renderer.scene.scene.set_sun_light(
        [0.707, -0.707, 0.0], [1.0, 1.0, 1.0], 1.5e5
    )
    renderer.scene.scene.enable_sun_light(True)

    renders_dir.mkdir(parents=True, exist_ok=True)

    fov_deg = 60.0
    up = [0.0, 1.0, 0.0]
    at = [0.0, 0.0, 0.0]

    for view_name, azim_deg in zip(VIEW_NAMES, VIEW_AZIMUTHS):
        azim = np.radians(azim_deg)
        elev = np.radians(ELEVATION_DEG)
        eye = [
            radius * np.cos(elev) * np.cos(azim),
            radius * np.sin(elev),
            radius * np.cos(elev) * np.sin(azim),
        ]
        renderer.setup_camera(fov_deg, at, eye, up)
        img = renderer.render_to_image()
        o3d.io.write_image(str(renders_dir / f"{view_name}.png"), img)

    del renderer


def render_views_trimesh(obj_path: Path, renders_dir: Path,
                         radius: float, render_size: int):
    """
    Fallback renderer using trimesh + pyrender (also needs EGL/display).
    Less accurate but works where open3d OffscreenRenderer fails.
    """
    import trimesh
    import pyrender
    import cv2

    mesh = trimesh.load(str(obj_path), force='mesh')
    verts = np.asarray(mesh.vertices, np.float64)
    center = verts.mean(axis=0)
    extent = np.linalg.norm(verts - center, axis=1).max()
    verts = (verts - center) / (extent + 1e-8)
    mesh.vertices = verts

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.radians(60.0))
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    r = pyrender.OffscreenRenderer(render_size, render_size)
    renders_dir.mkdir(parents=True, exist_ok=True)

    for view_name, azim_deg in zip(VIEW_NAMES, VIEW_AZIMUTHS):
        azim = np.radians(azim_deg)
        elev = np.radians(ELEVATION_DEG)
        eye = np.array([
            radius * np.cos(elev) * np.cos(azim),
            radius * np.sin(elev),
            radius * np.cos(elev) * np.sin(azim),
        ])
        at = np.zeros(3)
        up = np.array([0.0, 1.0, 0.0])
        z = (eye - at); z /= np.linalg.norm(z)
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = x;  cam_pose[:3, 1] = y
        cam_pose[:3, 2] = z;  cam_pose[:3, 3] = eye

        cam_node   = scene.add(camera, pose=cam_pose)
        light_node = scene.add(light,  pose=cam_pose)
        color, _   = r.render(scene)
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

        Image.fromarray(color).save(str(renders_dir / f"{view_name}.png"))

    r.delete()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glbs_dir",   required=True,
                        help="Directory containing *.glb files")
    parser.add_argument("--output_dir", required=True,
                        help="Output frame_XXXX/ directory")
    parser.add_argument("--radius",     type=float, default=1.5,
                        help="Camera radius (default: 1.5)")
    parser.add_argument("--render_size", type=int, default=1024,
                        help="Square render resolution (default: 1024)")
    args = parser.parse_args()

    glbs_dir   = Path(args.glbs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(glbs_dir.glob("*.glb"))
    if not glb_files:
        glb_files = sorted(glbs_dir.glob("*.GLB"))
    if not glb_files:
        raise RuntimeError(f"No .glb files found in {glbs_dir}")

    print(f"Found {len(glb_files)} GLBs — processing into {output_dir}")

    # Decide renderer
    try:
        import open3d as o3d
        o3d.visualization.rendering.OffscreenRenderer(32, 32)
        render_fn = render_views_open3d
        print("Renderer: open3d OffscreenRenderer")
    except Exception:
        try:
            import pyrender  # noqa
            render_fn = render_views_trimesh
            print("Renderer: trimesh + pyrender (open3d EGL unavailable)")
        except ImportError:
            raise RuntimeError(
                "Neither open3d OffscreenRenderer nor pyrender is available.\n"
                "Run on a GPU node, or install pyrender: pip install pyrender"
            )

    for idx, glb_path in enumerate(glb_files, 1):
        frame_name = f"frame_{idx:04d}"
        frame_dir  = output_dir / frame_name
        conv_dir   = frame_dir / "converted"
        rend_dir   = frame_dir / "renders"

        # Skip if already done
        if (conv_dir / "mesh.obj").exists() and (rend_dir / "front.png").exists():
            print(f"  {frame_name}: already exists, skipping")
            continue

        print(f"  [{idx}/{len(glb_files)}] {glb_path.name} → {frame_name}", flush=True)

        try:
            obj_path = glb_to_obj(glb_path, conv_dir)
            render_fn(obj_path, rend_dir, args.radius, args.render_size)
            print(f"    OK — {len(list(rend_dir.glob('*.png')))} renders")
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {glb_path.name} — remove partial dir to retry")
            continue

    n_ready = sum(
        1 for i in range(1, len(glb_files) + 1)
        if (output_dir / f"frame_{i:04d}" / "renders" / "front.png").exists()
    )
    print(f"\nDone — {n_ready}/{len(glb_files)} frames ready in {output_dir}")


if __name__ == "__main__":
    main()
