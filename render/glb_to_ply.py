"""
glb_to_ply.py
-------------
Convert a .glb (or .gltf / .obj) into a .ply carrying PER-VERTEX COLOURS, which is
what render_single_mesh.py can actually import.

WHY THIS EXISTS
  render_single_mesh.py (copied unmodified from multi_iSeg, which took it from
  Itai's tree) accepts only .obj and .ply — see its lines 83 and 86. There is no
  glTF importer. TRELLIS exports .glb. This bridges the two.

  PLY stores colour as a per-vertex attribute (red/green/blue on each vertex).
  A .glb usually stores colour as a UV-mapped texture image plus a material. So
  the conversion is not a container swap: the texture has to be SAMPLED at each
  vertex's UV coordinate and written onto that vertex.

WHAT THAT COSTS, STATED PLAINLY
  Baking a texture down to vertex colours is lossy, and the loss is bounded by the
  mesh's vertex density, not the texture's resolution. A 2048x2048 texture on a
  200k-vertex mesh has ~4.2M texels and only 200k places to put them, so fine
  detail between vertices is discarded. For a dense TRELLIS mesh (~215k vertices
  over a single object) the vertex spacing is far finer than the rendered pixel
  size, so this is visually near-lossless -- but it is not free, and on a coarse
  mesh it would be obvious. The script reports the ratio so you can judge.

  If the source already has vertex colours (which TRELLIS meshes do natively --
  colour is a per-vertex attribute, there are no UVs), the conversion is exact
  and nothing is resampled.

USAGE
  python render/glb_to_ply.py input.glb output.ply
  python render/glb_to_ply.py input.glb output.ply --report
  python render/glb_to_ply.py 'frames/*.glb' outdir/            # batch
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit('trimesh is required:  pip install trimesh')


def _flatten(loaded):
    """A .glb is a Scene; a .ply/.obj is usually a Trimesh. Return one Trimesh."""
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError('scene contains no geometry')
        # concatenate applies each node's transform, so multi-part assets survive
        mesh = trimesh.util.concatenate(
            [g.copy().apply_transform(loaded.graph[name][0])
             for name, g in loaded.geometry.items()]
        ) if len(loaded.geometry) > 1 else list(loaded.geometry.values())[0].copy()
        return mesh
    return loaded


def describe_colour(mesh):
    """What kind of colour does this mesh carry?"""
    v = mesh.visual
    if isinstance(v, trimesh.visual.ColorVisuals):
        if v.vertex_colors is not None and len(v.vertex_colors):
            return 'vertex_colors'
        if v.face_colors is not None and len(v.face_colors):
            return 'face_colors'
        return 'none'
    if isinstance(v, trimesh.visual.TextureVisuals):
        mat = getattr(v, 'material', None)
        img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        if img is not None:
            return f'texture {img.size[0]}x{img.size[1]}'
        base = getattr(mat, 'baseColorFactor', None)
        return 'flat_material' if base is not None else 'none'
    return 'unknown'


def to_vertex_colours(mesh, verbose=True):
    """
    Return an (N, 4) uint8 RGBA array, one row per vertex.

    Texture -> vertex colour goes through trimesh's to_color(), which samples the
    base-colour image at each vertex's UV. Meshes that already carry vertex
    colours are passed through untouched.
    """
    kind = describe_colour(mesh)
    if verbose:
        print(f'  source colour : {kind}')

    if kind == 'vertex_colors':
        if verbose:
            print('  -> already per-vertex; copied exactly, nothing resampled')
        return np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)

    if kind.startswith('texture') or kind == 'flat_material':
        conv = mesh.visual.to_color()
        vc = getattr(conv, 'vertex_colors', None)
        if vc is None or not len(vc):
            raise ValueError('texture -> vertex colour conversion produced nothing')
        vc = np.asarray(vc, dtype=np.uint8)
        if verbose and kind.startswith('texture'):
            w, h = (int(x) for x in kind.split()[1].split('x'))
            texels, verts = w * h, len(mesh.vertices)
            print(f'  -> sampled at {verts:,} vertices from {texels:,} texels '
                  f'({texels/max(verts,1):.1f} texels per vertex — detail finer '
                  f'than the vertex spacing is lost)')
        return vc

    if kind == 'face_colors':
        conv = mesh.visual.to_color()
        return np.asarray(conv.vertex_colors, dtype=np.uint8)

    if verbose:
        print('  -> no colour found; writing mid-grey so the render is not black')
    return np.tile(np.array([[128, 128, 128, 255]], np.uint8), (len(mesh.vertices), 1))


def normalize_to_unit(mesh, verbose=True, target=1.0):
    """
    Centre at the origin and scale the longest side to `target`.

    render_single_mesh.py's --object-scale / --camera-location defaults assume a
    roughly unit-sized object at the origin. Feed it a mesh spanning 76 units and
    you get a black wedge filling the frame — the object is real, it is just
    enormous and clipping the camera.

    TRELLIS meshes already live in [-0.5, 0.5] (see get_defomed_verts in
    utils_cube.py), so this is a no-op for our assets. It matters for arbitrary
    glb files.

    Applied to a COPY of the vertices; the source file is never touched.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if extent <= 0:
        return mesh, 1.0, centre
    s = target / extent
    mesh.vertices = (v - centre) * s
    if verbose:
        print(f'  normalised    : centred, longest side {extent:.4f} -> {target:.1f} '
              f'(scale {s:.5f})')
    return mesh, s, centre


def convert(src, dst, verbose=True, normalize=False):
    src, dst = Path(src), Path(dst)
    if verbose:
        print(f'[{src.name}]')
    mesh = _flatten(trimesh.load(str(src), process=False, force='mesh'
                                 if src.suffix.lower() in ('.ply', '.obj') else None))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'could not resolve {src} to a single mesh')

    vc = to_vertex_colours(mesh, verbose)
    if len(vc) != len(mesh.vertices):
        raise ValueError(f'{len(vc)} colours for {len(mesh.vertices)} vertices')

    if normalize:
        mesh, _, _ = normalize_to_unit(mesh, verbose)
    elif verbose:
        ext = float((np.asarray(mesh.vertices).max(axis=0)
                     - np.asarray(mesh.vertices).min(axis=0)).max())
        if ext > 4.0 or ext < 0.05:
            print(f'  WARNING       : longest side is {ext:.3f}. render_single_mesh.py '
                  f'assumes ~unit size; pass --normalize or the render will be '
                  f'clipped or invisible.')

    out = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                          vertex_colors=vc, process=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(dst))

    if verbose:
        b = out.bounds
        print(f'  vertices      : {len(out.vertices):,}   faces: {len(out.faces):,}')
        print(f'  bounds        : min {np.round(b[0], 4).tolist()}  '
              f'max {np.round(b[1], 4).tolist()}')
        print(f'  mean RGB      : {np.round(vc[:, :3].mean(axis=0), 1).tolist()}')
        print(f'  wrote         : {dst}  ({dst.stat().st_size/1e6:.1f} MB)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='input .glb/.gltf/.obj/.ply, or a glob for batch mode')
    ap.add_argument('dst', help='output .ply, or an output directory in batch mode')
    ap.add_argument('--normalize', action='store_true',
                    help='centre at origin and scale the longest side to 1.0. '
                         'render_single_mesh.py assumes a ~unit object; TRELLIS '
                         'meshes already are, arbitrary glb files are not.')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    matches = sorted(glob.glob(a.src)) if any(c in a.src for c in '*?[') else [a.src]
    if not matches:
        sys.exit(f'no files matched: {a.src}')

    if len(matches) > 1 or os.path.isdir(a.dst):
        outdir = Path(a.dst); outdir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(matches, 1):
            convert(m, outdir / (Path(m).stem + '.ply'), not a.quiet, a.normalize)
            if a.quiet and i % 25 == 0:
                print(f'  {i}/{len(matches)}', flush=True)
        print(f'[DONE] {len(matches)} file(s) -> {outdir}')
    else:
        convert(matches[0], a.dst, not a.quiet, a.normalize)
        print('[DONE]')


if __name__ == '__main__':
    main()
