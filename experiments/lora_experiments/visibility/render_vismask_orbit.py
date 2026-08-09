"""
render_vismask_orbit.py
-----------------------
Step 2 of Option B — apply the visibility mask at INFERENCE TIME and render the
360° turntable.  NO RETRAINING NEEDED for this script: it loads the existing v4
checkpoint and simply zeroes the colour delta on voxels the front camera never saw.

That is exact, not an approximation: the out_layer LoRA computes
    delta[v] = B @ A @ x[v]
purely from voxel v's own 96-dim feature, with no cross-voxel mixing.  So
multiplying delta by a per-voxel weight is a well-defined edit of the decoder
output — the visible voxels render bit-identically to v4.

Produces a 3-panel turntable:
    frozen TRELLIS  |  v4 LoRA (unmasked)  |  v4 LoRA + visibility mask

Modes
  --sweep angle : one SLaT frame, camera orbits 360°        (isolates the sticker)
  --sweep both  : frame advances 1..150 while camera orbits (the "360 video")

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
    experiments/lora_experiments/visibility/render_vismask_orbit.py \
    --run-id c85c888f --sweep angle --frame 75 --n-angles 60 \
    --mask-mode soft --elevation 15 \
    2>&1 | tee experiments/lora_experiments/visibility/logs/orbit_vismask_soft.log
"""

import sys, os, json, gc, math, argparse, subprocess
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'
_LORA = _ROOT / 'experiments' / 'lora_experiments'

sys.path.insert(0, str(_HERE))
from vis_mask import (add_mask_args, load_mask, mask_kwargs,   # noqa: E402
                      channel_mask)

ap = argparse.ArgumentParser()
ap.add_argument('--run-id',    default='c85c888f')
ap.add_argument('--run-dir',   default=None, type=Path)
ap.add_argument('--ckpt',      default=None, type=Path,
                help='explicit LoRA checkpoint (default: <run>/lora_ckpts/lora_best.pt)')
ap.add_argument('--out-dir',   default=None, type=Path)
ap.add_argument('--sweep',     default='angle', choices=['angle', 'both'])
ap.add_argument('--frame',     type=int, default=75,
                help="SLaT frame for --sweep angle")
ap.add_argument('--n-angles',  type=int, default=60)
ap.add_argument('--n-frames',  type=int, default=150)
ap.add_argument('--elevation', type=float, default=15.0)
ap.add_argument('--radius',    type=float, default=2.0)
ap.add_argument('--rank',      type=int, default=4)
ap.add_argument('--fps',       type=int, default=15)
add_mask_args(ap)
args = ap.parse_args()

RUN_DIR = args.run_dir or (_LORA / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}')
RUN_DIR = Path(RUN_DIR).resolve()
CKPT    = Path(args.ckpt) if args.ckpt else (RUN_DIR / 'lora_ckpts' / 'lora_best.pt')
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'

_tag    = f'{args.sweep}_{args.mask_mode}'
OUT_DIR = (args.out_dir or (_HERE / 'orbit_renders' / f'{args.run_id}_{_tag}')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_OUT_IN_DIM = 96
COLOR_START    = 53
COLOR_END      = 101
COLOR_DIM      = COLOR_END - COLOR_START
ABS_MIN        = -9.0
ABS_MAX        =  8.0


# ── nvdiffrast arch guard ─────────────────────────────────────────────────────
def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm{p.major}{p.minor}'
    arch_str = f'{p.major}.{p.minor}'
    local    = f'/tmp/nvdiffrast_{arch_tag}'
    print(f'[NVDIFF] GPU: {p.name}  {arch_tag}', flush=True)
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, local)
    try:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext(); del glctx
        print('[NVDIFF] OK', flush=True); return
    except Exception as e:
        print(f'[NVDIFF] FAILED: {e}', flush=True)
    if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')
    print(f'[NVDIFF] building for {arch_tag} -> {local}', flush=True)
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local,
             '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
            cwd=f'{src}/nvdiffrast', env=env, check=True)
    print('[NVDIFF] build done — restarting', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, INTRINSICS,
)

DEVICE = torch.device('cuda')


class OutLayerLoRA(nn.Module):
    """Identical to the v4 module — checkpoints load straight in."""
    def __init__(self, rank=4, in_dim=DEC_OUT_IN_DIM, color_dim=COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    """
    World-to-camera extrinsics for an orbit at azimuth theta, elevation el.
    theta=0 reproduces the confirmed front view EXTRINSICS (camera at (0,-r,0)).
    world_up=(0,0,-1) matches the original cam_y so the teapot stays upright.
    Same formula as experiments/decoder_same_geometry/debug_render_orbit_sweep.py.
    """
    theta = math.radians(theta_deg)
    el    = math.radians(elevation_deg)

    cx = radius * math.sin(theta) * math.cos(el)
    cy = -radius * math.cos(theta) * math.cos(el)
    cz = radius * math.sin(el)
    C  = torch.tensor([cx, cy, cz], dtype=torch.float32)

    world_up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cam_z    = -C / C.norm()
    cam_x    = torch.linalg.cross(world_up, cam_z); cam_x = cam_x / cam_x.norm()
    cam_y    = torch.linalg.cross(cam_z, cam_x);    cam_y = cam_y / cam_y.norm()

    R = torch.stack([cam_x, cam_y, cam_z], dim=0)
    t = R @ (-C)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3,  3] = t
    return E.to(DEVICE)


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def decode_triplet(dec_model, lora, w_vox, feats, coords, c_mask=None):
    """
    One decoder forward, three meshes:
      frozen        — no delta
      lora_unmasked — delta on every voxel            (= v4, the sticker)
      lora_masked   — delta * visibility weight       (= Option B)

    Only to_representation is re-run for the two LoRA variants; the expensive
    transformer + upsampler stack runs once.

    Passing w_vox=None disables masking (the masked mesh equals the unmasked one).
    """
    st       = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))
    captured = {}

    def _hook(mod, inp, out):
        captured['x']   = inp[0].feats     # (N_fine, 96)
        captured['out'] = out              # (N_fine, 101)

    with torch.no_grad():
        h             = dec_model.out_layer.register_forward_hook(_hook)
        frozen_meshes = dec_model(st)
        h.remove()
        frozen_mesh = filter_degenerate_faces(frozen_meshes[0])

        out_h = captured['out']
        delta = lora(captured['x'])                       # (N_fine, 48)
        if c_mask is not None:
            # applied to BOTH panels so the unmasked one stays a like-for-like
            # control: the only difference between them is the visibility mask
            delta = delta * c_mask.to(delta.dtype).unsqueeze(0)

        f_un = out_h.feats.clone()
        f_un[:, COLOR_START:COLOR_END] = (
            f_un[:, COLOR_START:COLOR_END] + delta
        ).clamp(ABS_MIN, ABS_MAX)
        mesh_un = filter_degenerate_faces(
            dec_model.to_representation(out_h.replace(f_un))[0])

        if w_vox is None:
            mesh_m = mesh_un
        else:
            assert w_vox.shape[0] == delta.shape[0], (
                f'mask length {w_vox.shape[0]} != N_fine {delta.shape[0]} — the '
                f'visibility npz was computed for a different SLaT structure'
            )
            wcol = w_vox.to(delta.dtype).unsqueeze(1)     # (N_fine, 1)
            f_m  = out_h.feats.clone()
            f_m[:, COLOR_START:COLOR_END] = (
                f_m[:, COLOR_START:COLOR_END] + delta * wcol
            ).clamp(ABS_MIN, ABS_MAX)
            mesh_m = filter_degenerate_faces(
                dec_model.to_representation(out_h.replace(f_m))[0])

    return frozen_mesh, mesh_un, mesh_m


def render_uint8(mesh, renderer, ext):
    res  = renderer.render(mesh, ext, INTRINSICS.to(DEVICE),
                           return_types=['color', 'mask'])
    m    = res['mask'].unsqueeze(0)
    col  = res['color'] * m + (1.0 - m)
    return (col.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


LABEL_H = 26


def strip(arrays, labels, colors):
    H, W = arrays[0].shape[:2]
    canvas = np.full((H + LABEL_H, W * len(arrays), 3), 15, dtype=np.uint8)
    pil    = Image.fromarray(canvas)
    draw   = ImageDraw.Draw(pil)
    for i, (arr, lbl, c) in enumerate(zip(arrays, labels, colors)):
        pil.paste(Image.fromarray(arr), (i * W, LABEL_H))
        draw.text((i * W + 8, 6), lbl, fill=c)
    return pil


def encode_video(pattern, out_path, fps):
    ff    = '/usr/bin/ffmpeg'
    probe = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
    if 'libx264' in probe.stdout:
        flags = ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
    elif 'libopenh264' in probe.stdout:
        flags = ['-c:v', 'libopenh264', '-b:v', '4M', '-pix_fmt', 'yuv420p']
    elif 'h264_nvenc' in probe.stdout:
        flags = ['-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '18', '-pix_fmt', 'yuv420p']
    else:
        flags = ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p']
    print(f'[FFMPEG] encoder: {flags[1]}', flush=True)
    subprocess.run([ff, '-y', '-framerate', str(fps), '-i', str(pattern),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *flags,
                    str(out_path)], check=True)
    print(f'[VIDEO] {out_path}', flush=True)


def main():
    print('=' * 72)
    print('Visibility-masked 360° turntable  (inference-time masking, no retrain)')
    print(f'  run dir  : {RUN_DIR}')
    print(f'  ckpt     : {CKPT}')
    print(f'  sweep    : {args.sweep}   angles={args.n_angles}  el={args.elevation}°')
    print(f'  out dir  : {OUT_DIR}')
    print('=' * 72, flush=True)

    assert SLAT_NPZ.exists(), f'missing SLaT cache: {SLAT_NPZ}'
    assert CKPT.exists(),     f'missing checkpoint: {CKPT}'

    print('[SLAT] loading cache...', flush=True)
    raw        = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr  = raw['slats']
    coords_t   = torch.from_numpy(raw['coords'].copy()).int()
    print(f'[SLAT] slats={slats_arr.shape}', flush=True)

    print('[MODEL] loading TRELLIS mesh decoder...', flush=True)
    pipe      = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
    dec_model = pipe.models['slat_decoder_mesh'].to(DEVICE).eval()
    for p in dec_model.parameters():
        p.requires_grad_(False)
    for name in list(pipe.models.keys()):
        if name != 'slat_decoder_mesh':
            try: pipe.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    lora = OutLayerLoRA(rank=args.rank).to(DEVICE)
    ck   = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    lora.load_state_dict(ck['lora_state'])
    lora.eval()
    print(f'[LORA] epoch={ck.get("epoch")}  best_psnr={ck.get("best_psnr", float("nan")):.3f}',
          flush=True)

    vm = load_mask(args)
    if vm is None:
        print('[MASK] mode=none — masked panel will equal the unmasked panel', flush=True)
        w_np = None
    else:
        # frame= is only consulted when agg == 'per-frame'; passing it always
        # keeps --mask-agg per-frame --sweep angle from raising.
        kw   = dict(mask_kwargs(args), frame=args.frame)
        print(f'[MASK] {vm.describe(**kw)}', flush=True)
        w_np = vm.weights(**kw)

    c_mask = channel_mask(args.delta_channels).to(DEVICE)
    print(f'[CHAN] delta_channels={args.delta_channels}  '
          f'active={int(c_mask.sum())}/48 '
          f'({"albedo + shading normals" if args.delta_channels == "all" else "albedo only"})',
          flush=True)

    renderer = make_renderer(DEVICE)
    frames_dir = OUT_DIR / 'frames'
    frames_dir.mkdir(exist_ok=True)

    n_steps = args.n_angles if args.sweep == 'angle' else args.n_frames
    print(f'\n[RENDER] {n_steps} steps → {frames_dir}', flush=True)

    mask_lbl = (f'+ vis mask ({args.mask_mode})' if vm is not None else '(no mask)')
    w_global = torch.from_numpy(w_np).to(DEVICE) if w_np is not None else None
    cached_meshes = None

    for idx in range(n_steps):
        theta = 360.0 * idx / n_steps
        fi    = args.frame if args.sweep == 'angle' else (idx + 1)

        if args.sweep == 'angle':
            # SLaT is fixed across the sweep — decode once, then only re-render
            if cached_meshes is None:
                feats = torch.from_numpy(slats_arr[fi - 1].copy()).float()
                cached_meshes = decode_triplet(
                    dec_model, lora, w_global, feats, coords_t, c_mask)
            m_frozen, m_un, m_mask = cached_meshes
        else:
            feats = torch.from_numpy(slats_arr[fi - 1].copy()).float()
            if vm is not None and args.mask_agg == 'per-frame':
                w_vox = torch.from_numpy(
                    vm.weights(**dict(mask_kwargs(args), frame=fi))).to(DEVICE)
            else:
                w_vox = w_global
            m_frozen, m_un, m_mask = decode_triplet(
                dec_model, lora, w_vox, feats, coords_t, c_mask)

        ext = orbit_extrinsics(theta, args.elevation, args.radius)
        a_frozen = render_uint8(m_frozen, renderer, ext)
        a_un     = render_uint8(m_un,     renderer, ext)
        a_mask   = render_uint8(m_mask,   renderer, ext)

        ang = f'{theta:5.0f}°'
        img = strip(
            [a_frozen, a_un, a_mask],
            [f'frozen TRELLIS   {ang}',
             f'v4 LoRA (unmasked)   {ang}',
             f'v4 LoRA {mask_lbl}   {ang}'],
            [(200, 80, 80), (240, 190, 90), (90, 190, 255)],
        )
        img.save(frames_dir / f'frame_{idx + 1:04d}.png')
        Image.fromarray(a_mask).save(frames_dir / f'masked_{idx + 1:04d}.png')

        if idx % 10 == 0 or idx == n_steps - 1:
            print(f'  {idx + 1}/{n_steps}  θ={theta:.0f}°  f{fi:03d}', flush=True)

        if args.sweep != 'angle':
            del m_frozen, m_un, m_mask
            gc.collect(); torch.cuda.empty_cache()

    dec_model.cpu(); gc.collect(); torch.cuda.empty_cache()
    print('[RENDER] done.', flush=True)

    encode_video(frames_dir / 'frame_%04d.png',
                 OUT_DIR / f'orbit_compare_{_tag}.mp4', args.fps)
    encode_video(frames_dir / 'masked_%04d.png',
                 OUT_DIR / f'orbit_masked_only_{_tag}.mp4', args.fps)

    json.dump({
        'run_dir': str(RUN_DIR), 'ckpt': str(CKPT), 'sweep': args.sweep,
        'frame': args.frame, 'n_angles': args.n_angles, 'n_frames': args.n_frames,
        'elevation': args.elevation, 'radius': args.radius,
        'mask': mask_kwargs(args), 'delta_channels': args.delta_channels,
        'mask_stats': (vm.describe(**dict(mask_kwargs(args), frame=args.frame))
                       if vm is not None else 'none'),
    }, open(OUT_DIR / 'render_config.json', 'w'), indent=2)
    print(f'[DONE] {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
