"""
compute_visibility_mask.py
--------------------------
Step 1 of Option B (visibility masking).

Computes, for every fine voxel at dec_model.out_layer, HOW MUCH that voxel's
color channels influence the FRONT-CAMERA render.

Method — gradient probe (not geometric projection):
  For each frame:
    1. decode the frozen SLaT → mesh, but intercept out_layer and replace the
       color slice [53:101] with a leaf tensor that requires grad
    2. render from EXTRINSICS (the confirmed front view)
    3. backward on (masked pixel sum) * LOSS_SCALE
    4. per-voxel score = ||d(render) / d(color_feats_voxel)||_2   over the 48 dims

  score > 0  <=>  the front render actually depends on this voxel
  score == 0 <=>  this voxel was NEVER seen by the training camera, so any color
                  delta v4 learned for it is pure noise — this is the sticker.

Why gradient and not "project voxel centers + depth test": the mesh is produced
by flexicubes, which interpolates vertex colors from a neighbourhood of voxels.
The gradient tells us exactly which voxels the rasterized image is a function of,
including the sub-surface ones a naive depth test would wrongly discard.

LOSS_SCALE is required: dec_mesh runs in fp16 and small gradients flush to zero
without it (same root cause as the step8 zero-gradient bug).

Scores are stored as float32, NOT float16. The scores are gradient norms divided
by LOSS_SCALE=4096, so a voxel with grad_norm 1e-4 lands at 2.4e-8 — below the
float16 smallest subnormal (6e-8) and would silently flush to zero. Those are
exactly the grazing-angle voxels the soft ramp exists to fade, so float16 here
would quietly turn the soft mask back into a hard one.

Output: <out>/visibility.npz
  scores    float32 (N_FRAMES, N_fine)  raw per-voxel gradient norms
  coords    int32   (N_coarse, 4)       the shared coarse coords (provenance)
  frames    int32   (N_FRAMES,)         frame indices, 1-based
  meta      json string                 run_id, N_fine, extrinsics, etc.

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
    experiments/lora_experiments/visibility/compute_visibility_mask.py \
    --run-id c85c888f \
    2>&1 | tee experiments/lora_experiments/visibility/logs/compute_visibility.log
"""

import sys, os, json, gc, math, argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--run-id',   default='c85c888f',
                help='v4 run id — used only to locate slat_cache.npz')
ap.add_argument('--run-dir',  default=None, type=Path,
                help='explicit run dir (overrides --run-id)')
ap.add_argument('--out',      default=None, type=Path,
                help='output dir (default: <this dir>/masks)')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--stride',   type=int, default=1,
                help='probe every Nth frame; 1 = all frames')
ap.add_argument('--smoke',    action='store_true',
                help='probe 3 frames only')
args = ap.parse_args()

RUN_DIR = args.run_dir or (_LORA / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}')
RUN_DIR = Path(RUN_DIR).resolve()
OUT_DIR = (args.out or (_HERE / 'masks')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLAT_NPZ    = RUN_DIR / 'slat_cache.npz'
N_FRAMES    = args.n_frames
COLOR_START = 53
COLOR_END   = 101
COLOR_DIM   = COLOR_END - COLOR_START   # 48
LOSS_SCALE  = 4096.0


# ── nvdiffrast arch guard ─────────────────────────────────────────────────────
def _ensure_nvdiffrast():
    import subprocess, torch
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
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    print('[NVDIFF] build done — restarting', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import (
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS,
)

DEVICE = torch.device('cuda')


def filter_degenerate_faces(mesh, min_area=1e-6):
    v, f = mesh.vertices, mesh.faces
    e1   = v[f[:, 1]] - v[f[:, 0]]
    e2   = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
    mesh.faces = f[area > min_area]
    return mesh


def probe_frame(dec_model, renderer, slat_norm):
    """
    One forward + backward. Returns (score (N_fine,), N_fine).

    The hook detaches geometry [0:53] (we do not need its gradient and it saves
    memory) and swaps the color slice for a leaf tensor whose .grad we read back.
    """
    captured = {}

    def _hook(mod, inp, out):
        feats = out.feats
        col   = feats[:, COLOR_START:COLOR_END].detach().clone().requires_grad_(True)
        captured['col'] = col
        new_feats = torch.cat([
            feats[:, :COLOR_START].detach(),
            col,
            feats[:, COLOR_END:].detach(),
        ], dim=1)
        return out.replace(new_feats)

    with torch.enable_grad():
        h      = dec_model.out_layer.register_forward_hook(_hook)
        meshes = dec_model(slat_norm)
        h.remove()

        mesh = filter_degenerate_faces(meshes[0])
        ext  = EXTRINSICS.to(DEVICE)
        intr = INTRINSICS.to(DEVICE)
        res  = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
        m    = res['mask'].unsqueeze(0)
        color = res['color'] * m          # background contributes nothing

        # scalar objective whose gradient w.r.t. each voxel's color is nonzero
        # iff that voxel appears in the front image
        obj = color.sum()
        (obj * LOSS_SCALE).backward()

    col   = captured['col']
    grad  = col.grad
    if grad is None:
        raise RuntimeError('no gradient reached out_layer color channels — '
                           'check LOSS_SCALE / fp16 flush-to-zero')
    score = grad.detach().float().norm(dim=1) / LOSS_SCALE   # (N_fine,)

    n_fine = score.shape[0]
    del captured, meshes, mesh, res, color, obj, grad
    gc.collect(); torch.cuda.empty_cache()
    return score.cpu(), n_fine


def main():
    print('=' * 72)
    print('Front-camera visibility probe (Option B, step 1)')
    print(f'  run dir   : {RUN_DIR}')
    print(f'  slat cache: {SLAT_NPZ}')
    print(f'  out dir   : {OUT_DIR}')
    print(f'  loss scale: {LOSS_SCALE:.0f}')
    print('=' * 72, flush=True)

    assert SLAT_NPZ.exists(), f'missing SLaT cache: {SLAT_NPZ}'

    print('[SLAT] loading cache...', flush=True)
    raw        = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr  = raw['slats']    # (150, 7301, 8) — already normalized
    coords_arr = raw['coords']   # (7301, 4)
    coords_t   = torch.from_numpy(coords_arr.copy()).int()
    print(f'[SLAT] slats={slats_arr.shape}  coords={coords_arr.shape}', flush=True)

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

    renderer = make_renderer(DEVICE)

    frames = list(range(1, N_FRAMES + 1, args.stride))
    if args.smoke:
        frames = [1, 75, 150]
    print(f'\n[PROBE] {len(frames)} frames', flush=True)

    scores  = None
    n_fine  = None
    for k, fi in enumerate(frames):
        feats     = torch.from_numpy(slats_arr[fi - 1].copy()).float()
        slat_norm = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords_t.to(DEVICE))
        s, nf     = probe_frame(dec_model, renderer, slat_norm)

        if scores is None:
            n_fine = nf
            scores = np.zeros((len(frames), n_fine), dtype=np.float32)
            print(f'[PROBE] N_fine = {n_fine:,} voxels at out_layer', flush=True)
        assert nf == n_fine, (
            f'N_fine changed between frames ({nf} != {n_fine}); the coarse coords '
            f'are supposed to be shared across all frames'
        )
        scores[k] = s.numpy()

        if (k + 1) % 10 == 0 or k == len(frames) - 1:
            vis_frac = float((s > 0).float().mean())
            print(f'  {k+1}/{len(frames)}  f{fi:03d}  '
                  f'visible={vis_frac*100:.2f}%  max_grad={float(s.max()):.3e}',
                  flush=True)

        del slat_norm, feats, s
        gc.collect(); torch.cuda.empty_cache()

    # ── summary statistics ────────────────────────────────────────────────────
    nz        = scores > 0
    per_frame = nz.mean(axis=1)
    ever      = nz.any(axis=0)
    always    = nz.all(axis=0)

    print('\n' + '=' * 72)
    print('[STATS]')
    print(f'  N_fine voxels          : {n_fine:,}')
    print(f'  visible per frame      : {per_frame.mean()*100:.2f}% '
          f'(min {per_frame.min()*100:.2f}%, max {per_frame.max()*100:.2f}%)')
    print(f'  visible in ANY frame   : {ever.mean()*100:.2f}%  ({int(ever.sum()):,} voxels)')
    print(f'  visible in EVERY frame : {always.mean()*100:.2f}%  ({int(always.sum()):,} voxels)')
    print(f'  NEVER visible          : {(1-ever.mean())*100:.2f}%  '
          f'({int((~ever).sum()):,} voxels)  <-- these carry the sticker artifact')
    pos = scores[nz]
    if pos.size:
        print(f'  score range (nonzero)  : min={pos.min():.3e}  '
              f'median={np.median(pos):.3e}  max={pos.max():.3e}')
        lost = int((pos < np.finfo(np.float16).smallest_subnormal).sum())
        print(f'  would underflow float16: {lost:,} / {pos.size:,} nonzero entries '
              f'({100.0*lost/pos.size:.2f}%) — stored as float32, none lost')
    print('=' * 72, flush=True)

    meta = {
        'run_dir'    : str(RUN_DIR),
        'run_id'     : args.run_id,
        'n_fine'     : int(n_fine),
        'n_frames'   : len(frames),
        'stride'     : args.stride,
        'color_start': COLOR_START,
        'color_end'  : COLOR_END,
        'loss_scale' : LOSS_SCALE,
        'render_res' : RENDER_RES,
        'extrinsics' : EXTRINSICS.tolist(),
        'frac_visible_any'   : float(ever.mean()),
        'frac_visible_all'   : float(always.mean()),
        'frac_visible_mean'  : float(per_frame.mean()),
    }

    out_npz = OUT_DIR / 'visibility.npz'
    np.savez_compressed(
        out_npz,
        scores = scores,
        coords = coords_arr.astype(np.int32),
        frames = np.array(frames, dtype=np.int32),
        meta   = json.dumps(meta),
    )
    print(f'\n[SAVE] {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB)', flush=True)
    json.dump(meta, open(OUT_DIR / 'visibility_meta.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "visibility_meta.json"}', flush=True)
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
