"""
diagnose_geometry_vs_appearance.py
----------------------------------
Answers one question with numbers instead of impressions:

    is the "sticker" a GEOMETRY problem or an APPEARANCE problem?

For a given LoRA checkpoint it decodes the frozen mesh and the LoRA mesh from
the same SLaT, then reports:

  A. GEOMETRY IDENTITY
       max |vertices_frozen - vertices_lora|,  face-array equality, vertex/face
       counts.  For an out_layer colour LoRA these must be EXACTLY equal: mesh
       vertex positions come only from sdf[0:8] and deform[8:32]
       (cube2mesh.py: x_nx3 = get_defomed_verts(reg_v, deform_d, res)), and the
       LoRA writes only [53:101].  A nonzero number here means the checkpoint
       is NOT geometry-preserving — e.g. an attention-block LoRA, which edits
       768-dim features BEFORE out_layer and therefore decodes into geometry
       as well as appearance.

  B. SILHOUETTE IDENTITY PER ORBIT ANGLE
       IoU between the frozen and LoRA render masks at each azimuth.  If the
       geometry is untouched this is 1.000 at every angle.  If it drops, the
       surface moved and you can see from which directions.

  C. APPEARANCE DELTA PER ORBIT ANGLE
       mean |colour_frozen - colour_lora| inside the shared mask.  THIS is the
       sticker, quantified: it peaks at the training view (theta=0) and falls
       off with angle.  The curve is directly usable as a paper figure — it
       shows the texture is a function of viewing angle, which is the honest
       statement of the limitation.

Interpretation cheat-sheet:
   A == 0 and B == 1 everywhere  ->  geometry is fine; the sticker is purely an
                                     under-determined-supervision artifact
                                     (front-view loss constrains the 2D
                                     projection, not the 3D colour field)
   A  > 0 or  B  < 1             ->  that checkpoint really is deforming the
                                     surface; the geometry complaint is real

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
    experiments/lora_experiments/visibility/diagnose_geometry_vs_appearance.py \
    --run-id c85c888f --frame 75 --n-angles 36 \
    2>&1 | tee experiments/lora_experiments/visibility/logs/diagnose_geom_c85c888f.log
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

sys.path.insert(0, str(_HERE))
from vis_mask import channel_mask   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run-id',   default='c85c888f')
ap.add_argument('--run-dir',  default=None, type=Path)
ap.add_argument('--ckpt',     default=None, type=Path)
ap.add_argument('--out-dir',  default=None, type=Path)
ap.add_argument('--frame',    type=int,   default=75)
ap.add_argument('--n-angles', type=int,   default=36)
ap.add_argument('--elevation', type=float, default=15.0)
ap.add_argument('--radius',   type=float, default=2.0)
ap.add_argument('--rank',     type=int,   default=4)
ap.add_argument('--delta-channels', default='all', choices=['all', 'rgb'])
args = ap.parse_args()

RUN_DIR  = args.run_dir or (_LORA / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}')
RUN_DIR  = Path(RUN_DIR).resolve()
CKPT     = Path(args.ckpt) if args.ckpt else (RUN_DIR / 'lora_ckpts' / 'lora_best.pt')
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
OUT_DIR  = (args.out_dir or (_HERE / 'diagnostics' / args.run_id)).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_OUT_IN_DIM = 96
COLOR_START, COLOR_END = 53, 101
COLOR_DIM = COLOR_END - COLOR_START
ABS_MIN, ABS_MAX = -9.0, 8.0


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
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local, '--no-build-isolation',
             '--no-cache-dir', '--no-deps', '-q'],
            cwd=f'{src}/nvdiffrast', env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp
from step8_decode_render.decode_render import make_renderer, INTRINSICS

DEVICE = torch.device('cuda')


class OutLayerLoRA(nn.Module):
    def __init__(self, rank=4, in_dim=DEC_OUT_IN_DIM, color_dim=COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def orbit_extrinsics(theta_deg, elevation_deg=0.0, radius=2.0):
    theta = math.radians(theta_deg); el = math.radians(elevation_deg)
    C = torch.tensor([radius * math.sin(theta) * math.cos(el),
                      -radius * math.cos(theta) * math.cos(el),
                      radius * math.sin(el)], dtype=torch.float32)
    world_up = torch.tensor([0., 0., -1.], dtype=torch.float32)
    cam_z = -C / C.norm()
    cam_x = torch.linalg.cross(world_up, cam_z); cam_x = cam_x / cam_x.norm()
    cam_y = torch.linalg.cross(cam_z, cam_x);    cam_y = cam_y / cam_y.norm()
    R = torch.stack([cam_x, cam_y, cam_z], dim=0)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3,  3] = R @ (-C)
    return E.to(DEVICE)


def decode_pair(dec_model, lora, feats, coords, c_mask):
    """Frozen mesh and LoRA mesh from ONE decoder forward. No face filtering —
    we want the raw extractor output so the comparison is not confounded."""
    st = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE))
    captured = {}

    def _hook(mod, inp, out):
        captured['x'] = inp[0].feats
        captured['out'] = out

    with torch.no_grad():
        h = dec_model.out_layer.register_forward_hook(_hook)
        frozen_meshes = dec_model(st)
        h.remove()

        out_h = captured['out']
        delta = lora(captured['x'])
        if c_mask is not None:
            delta = delta * c_mask.to(delta.dtype).unsqueeze(0)

        f = out_h.feats.clone()
        f[:, COLOR_START:COLOR_END] = (
            f[:, COLOR_START:COLOR_END] + delta).clamp(ABS_MIN, ABS_MAX)
        lora_meshes = dec_model.to_representation(out_h.replace(f))

    return frozen_meshes[0], lora_meshes[0], delta


def render(mesh, renderer, ext):
    res = renderer.render(mesh, ext, INTRINSICS.to(DEVICE),
                          return_types=['color', 'mask'])
    m = res['mask']
    col = res['color'] * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))
    return col.detach().clamp(0, 1), m.detach()


def main():
    print('=' * 72)
    print('Geometry vs appearance diagnostic')
    print(f'  run   : {RUN_DIR.name}')
    print(f'  ckpt  : {CKPT}')
    print(f'  frame : {args.frame}   angles: {args.n_angles}   el: {args.elevation}')
    print('=' * 72, flush=True)

    assert SLAT_NPZ.exists(), f'missing SLaT cache: {SLAT_NPZ}'
    assert CKPT.exists(),     f'missing checkpoint: {CKPT}'

    raw       = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr = raw['slats']
    coords_t  = torch.from_numpy(raw['coords'].copy()).int()

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
    print(f'[LORA] epoch={ck.get("epoch")}  ||B||={lora.B.float().norm().item():.4f}',
          flush=True)

    c_mask   = channel_mask(args.delta_channels).to(DEVICE)
    renderer = make_renderer(DEVICE)

    feats = torch.from_numpy(slats_arr[args.frame - 1].copy()).float()
    m_frz, m_lora, delta = decode_pair(dec_model, lora, feats, coords_t, c_mask)

    # ── A. geometry identity ──────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('[A] GEOMETRY IDENTITY')
    print('=' * 72)
    vf, vl = m_frz.vertices, m_lora.vertices
    ff, fl = m_frz.faces,    m_lora.faces
    print(f'  vertices : frozen {tuple(vf.shape)}   lora {tuple(vl.shape)}')
    print(f'  faces    : frozen {tuple(ff.shape)}   lora {tuple(fl.shape)}')

    same_shape = (vf.shape == vl.shape) and (ff.shape == fl.shape)
    if same_shape:
        dv = (vf.float() - vl.float()).abs()
        faces_equal = bool(torch.equal(ff, fl))
        print(f'  max |vertex delta|      : {dv.max().item():.6e}')
        print(f'  mean |vertex delta|     : {dv.mean().item():.6e}')
        print(f'  faces bit-identical     : {faces_equal}')
        geom_untouched = (dv.max().item() == 0.0) and faces_equal
    else:
        print('  !! vertex/face COUNTS differ — geometry definitely changed')
        geom_untouched = False

    print()
    if geom_untouched:
        print('  VERDICT: geometry is BIT-IDENTICAL. This checkpoint does not')
        print('           touch the mesh at all. Any "wrapping" or "sticker"')
        print('           look is an APPEARANCE / supervision artifact.')
    else:
        print('  VERDICT: geometry CHANGED. This checkpoint edits the surface —')
        print('           expect an attention/MLP-block LoRA, which decodes into')
        print('           geometry [0:53] as well as appearance [53:101].')

    print(f'\n  colour delta stats: max|d|={delta.abs().max().item():.4f}  '
          f'mean|d|={delta.abs().mean().item():.6f}  '
          f'nonzero voxels={int((delta.abs().sum(1) > 0).sum()):,}/{delta.shape[0]:,}')

    # ── B/C. per-angle silhouette + appearance ────────────────────────────────
    print('\n' + '=' * 72)
    print('[B/C] PER-ANGLE SILHOUETTE IoU AND APPEARANCE DELTA')
    print('=' * 72)
    print(f'{"theta":>7} {"mask IoU":>10} {"mean|dRGB|":>12} {"max|dRGB|":>11}')

    angles, ious, dmeans, dmaxs = [], [], [], []
    frames_dir = OUT_DIR / 'frames'
    frames_dir.mkdir(exist_ok=True)

    for i in range(args.n_angles):
        th  = 360.0 * i / args.n_angles
        ext = orbit_extrinsics(th, args.elevation, args.radius)
        c_f, k_f = render(m_frz,  renderer, ext)
        c_l, k_l = render(m_lora, renderer, ext)

        inter = (k_f * k_l).sum().item()
        union = ((k_f + k_l) > 0).float().sum().item()
        iou   = inter / union if union > 0 else 1.0

        shared = (k_f * k_l).unsqueeze(0)
        n      = shared.sum().item() * 3 + 1e-8
        dmean  = ((c_f - c_l).abs() * shared).sum().item() / n
        dmax   = ((c_f - c_l).abs() * shared).max().item()

        angles.append(th); ious.append(iou); dmeans.append(dmean); dmaxs.append(dmax)
        print(f'{th:7.1f} {iou:10.6f} {dmean:12.6f} {dmax:11.6f}', flush=True)

        side = torch.cat([c_f, c_l, (c_f - c_l).abs().clamp(0, 1)], dim=2)
        arr  = (side.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(frames_dir / f'cmp_{i + 1:04d}.png')

        del c_f, c_l, k_f, k_l
        gc.collect(); torch.cuda.empty_cache()

    iou_min = float(np.min(ious))
    print(f'\n  min silhouette IoU over all angles: {iou_min:.6f}')
    if iou_min >= 0.9999:
        print('  -> silhouettes identical at every angle: geometry confirmed frozen.')
    else:
        worst = angles[int(np.argmin(ious))]
        print(f'  -> silhouette differs (worst at theta={worst:.0f}deg): '
              f'the surface moved.')

    front_d = dmeans[0]
    back_i  = int(np.argmin(np.abs(np.array(angles) - 180.0)))
    back_d  = dmeans[back_i]
    ratio   = front_d / (back_d + 1e-12)
    print(f'\n  appearance delta at theta=0   (training view): {front_d:.6f}')
    print(f'  appearance delta at theta=180 (opposite)     : {back_d:.6f}')
    print(f'  front/back ratio                             : {ratio:.2f}x')
    print('  A large ratio IS the sticker, quantified: the edit is a function of')
    print('  viewing angle because the supervision only ever constrained one view.')

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.suptitle(f'Geometry vs appearance — {RUN_DIR.name}  frame {args.frame}',
                 fontsize=11)
    axes[0].plot(angles, ious, 'o-', color='#61afef', lw=2, ms=4)
    axes[0].axhline(1.0, color='#98c379', ls='--', lw=1, label='identical geometry')
    axes[0].set_xlabel('azimuth (deg)'); axes[0].set_ylabel('silhouette IoU')
    axes[0].set_title('B. geometry: frozen vs LoRA silhouette')
    axes[0].set_ylim(min(0.95, iou_min - 0.01), 1.005)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(angles, dmeans, 'o-', color='#e06c75', lw=2, ms=4, label='mean |dRGB|')
    axes[1].plot(angles, dmaxs,  's-', color='#d19a66', lw=1.5, ms=3, alpha=0.7,
                 label='max |dRGB|')
    axes[1].axvline(0.0, color='#98c379', ls='--', lw=1, label='training view')
    axes[1].set_xlabel('azimuth (deg)'); axes[1].set_ylabel('|frozen - LoRA|')
    axes[1].set_title('C. appearance: edit magnitude vs viewing angle')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'geometry_vs_appearance.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'\n[SAVE] {OUT_DIR / "geometry_vs_appearance.png"}')

    json.dump({
        'run_dir': str(RUN_DIR), 'ckpt': str(CKPT), 'frame': args.frame,
        'delta_channels': args.delta_channels,
        'geometry_untouched': bool(geom_untouched),
        'max_vertex_delta': (float(dv.max()) if same_shape else None),
        'faces_identical': (bool(faces_equal) if same_shape else False),
        'min_silhouette_iou': iou_min,
        'angles': angles, 'iou': ious,
        'appearance_delta_mean': dmeans, 'appearance_delta_max': dmaxs,
        'front_over_back_ratio': float(ratio),
    }, open(OUT_DIR / 'diagnosis.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "diagnosis.json"}')
    print(f'[SAVE] {frames_dir}/cmp_*.png  (frozen | lora | abs diff)')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
