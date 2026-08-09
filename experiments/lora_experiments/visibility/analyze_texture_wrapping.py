"""
analyze_texture_wrapping.py
---------------------------
Proves (or refutes) the mechanism behind "the texture wraps around the mesh".

HYPOTHESIS
  The out_layer LoRA computes  delta[v] = B @ A @ x[v]  from voxel v's own
  96-d feature. It has NO positional input. It is fitted using gradient from
  front-visible voxels only, but at inference it is applied to EVERY voxel.
  Because the object is roughly symmetric, never-seen voxels carry features
  similar to seen ones, so they receive a similar delta — the appearance edit
  generalises around the object by feature similarity. That is the wrapping.

WHAT IT MEASURES
  Voxels are split by the visibility probe into three groups:
      ALWAYS   visible in every probed frame
      SOMETIMES visible in some frames
      NEVER    visible in any frame   <- these are never supervised

  1. DELTA MAGNITUDE PER GROUP
       mean/median ||delta|| in each group, and the ratio NEVER / ALWAYS.
         ratio ~ 1   -> the edit is applied just as strongly to unsupervised
                        geometry as to supervised geometry. WRAPPING CONFIRMED.
         ratio ~ 0   -> the edit is confined to what the camera saw; the
                        artifact is a front decal, not wrapping.

  2. FEATURE-SIMILARITY COUPLING
       For NEVER-visible voxels, the max cosine similarity of their 96-d
       feature to the ALWAYS-visible set, correlated against ||delta||.
       A positive correlation is the smoking gun: unseen voxels that "look
       like" seen voxels inherit their appearance edit. That is the mechanism,
       not a coincidence.

  3. WHAT THE MASK RECOVERS
       Fraction of total applied delta energy that lands on NEVER-visible
       voxels — i.e. exactly how much of the edit visibility masking removes.

REQUIRES
  masks/visibility.npz  (run compute_visibility_mask.py first)

Usage:
  cd /net/projects/ranalab/rajhansini/TRELLIS
  export SPCONV_ALGO=native; export ATTN_BACKEND=xformers
  /net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python -u \
    experiments/lora_experiments/visibility/analyze_texture_wrapping.py \
    --run-id c85c888f --frame 75 \
    2>&1 | tee experiments/lora_experiments/visibility/logs/wrapping_c85c888f.log
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
from vis_mask import channel_mask, default_mask_npz   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run-id',   default='c85c888f')
ap.add_argument('--run-dir',  default=None, type=Path)
ap.add_argument('--ckpt',     default=None, type=Path)
ap.add_argument('--mask-npz', default=None, type=Path)
ap.add_argument('--out-dir',  default=None, type=Path)
ap.add_argument('--frame',    type=int, default=75)
ap.add_argument('--rank',     type=int, default=4)
ap.add_argument('--delta-channels', default='all', choices=['all', 'rgb'])
ap.add_argument('--n-sample', type=int, default=4000,
                help='NEVER-visible voxels sampled for the similarity test')
ap.add_argument('--n-ref',    type=int, default=20000,
                help='ALWAYS-visible voxels used as the reference set')
args = ap.parse_args()

RUN_DIR  = args.run_dir or (_LORA / 'runs' / f'rung5_colonly_outlayer_r4_s6_{args.run_id}')
RUN_DIR  = Path(RUN_DIR).resolve()
CKPT     = Path(args.ckpt) if args.ckpt else (RUN_DIR / 'lora_ckpts' / 'lora_best.pt')
SLAT_NPZ = RUN_DIR / 'slat_cache.npz'
MASK_NPZ = Path(args.mask_npz) if args.mask_npz else default_mask_npz()
OUT_DIR  = (args.out_dir or (_HERE / 'diagnostics' / f'{args.run_id}_wrapping')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_OUT_IN_DIM = 96
COLOR_START, COLOR_END = 53, 101
COLOR_DIM = COLOR_END - COLOR_START


def _ensure_nvdiffrast():
    import subprocess as _sp, torch
    p = torch.cuda.get_device_properties(0)
    arch_tag, arch_str = f'sm{p.major}{p.minor}', f'{p.major}.{p.minor}'
    local = f'/tmp/nvdiffrast_{arch_tag}'
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
        raise RuntimeError(f'nvdiffrast still broken for {arch_tag}')
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    _sp.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
             f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    _sp.run([pip, 'install', '.', '--target', local, '--no-build-isolation',
             '--no-cache-dir', '--no-deps', '-q'], cwd=f'{src}/nvdiffrast',
            env=env, check=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

DEVICE = torch.device('cuda')


class OutLayerLoRA(nn.Module):
    def __init__(self, rank=4, in_dim=DEC_OUT_IN_DIM, color_dim=COLOR_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(color_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def capture_features_and_delta(dec_model, lora, feats, coords, c_mask):
    """Run the decoder once; grab the 96-d pre-out_layer features and the delta."""
    st, cap = sp.SparseTensor(feats=feats.to(DEVICE), coords=coords.to(DEVICE)), {}

    def _hook(mod, inp, out):
        cap['x'] = inp[0].feats.detach()

    with torch.no_grad():
        h = dec_model.out_layer.register_forward_hook(_hook)
        dec_model(st)
        h.remove()
        x = cap['x'].float()
        delta = lora(x)
        if c_mask is not None:
            delta = delta * c_mask.to(delta.dtype).unsqueeze(0)
    return x, delta


def summarize(name, v):
    if v.numel() == 0:
        return f'  {name:<10} (empty)'
    return (f'  {name:<10} n={v.numel():>9,}  mean={v.mean():.6f}  '
            f'median={v.median():.6f}  p90={v.quantile(0.90):.6f}  '
            f'max={v.max():.6f}')


def main():
    print('=' * 74)
    print('Texture-wrapping mechanism analysis')
    print(f'  run   : {RUN_DIR.name}')
    print(f'  ckpt  : {CKPT}')
    print(f'  mask  : {MASK_NPZ}')
    print(f'  frame : {args.frame}   channels: {args.delta_channels}')
    print('=' * 74, flush=True)

    for p, what in ((SLAT_NPZ, 'SLaT cache'), (CKPT, 'checkpoint'),
                    (MASK_NPZ, 'visibility npz')):
        assert p.exists(), f'missing {what}: {p}\nrun compute_visibility_mask.py first'

    raw       = np.load(SLAT_NPZ, allow_pickle=True)
    slats_arr = raw['slats']
    coords_t  = torch.from_numpy(raw['coords'].copy()).int()

    vz     = np.load(MASK_NPZ, allow_pickle=True)
    scores = np.asarray(vz['scores'], dtype=np.float32)      # (F, N_fine)
    nz     = scores > 0
    always = torch.from_numpy(nz.all(axis=0)).to(DEVICE)
    ever   = torch.from_numpy(nz.any(axis=0)).to(DEVICE)
    never  = ~ever
    somet  = ever & ~always

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

    c_mask = channel_mask(args.delta_channels).to(DEVICE)
    feats  = torch.from_numpy(slats_arr[args.frame - 1].copy()).float()
    x, delta = capture_features_and_delta(dec_model, lora, feats, coords_t, c_mask)

    assert x.shape[0] == scores.shape[1], (
        f'N_fine mismatch: decoder {x.shape[0]} vs mask {scores.shape[1]}')

    dn = delta.norm(dim=1)                                    # (N_fine,)

    # ── 1. delta magnitude per visibility group ──────────────────────────────
    print('\n' + '=' * 74)
    print('[1] DELTA MAGNITUDE BY VISIBILITY GROUP')
    print('=' * 74)
    print(f'  voxels: ALWAYS={int(always.sum()):,}  SOMETIMES={int(somet.sum()):,}  '
          f'NEVER={int(never.sum()):,}  total={dn.numel():,}')
    print(summarize('ALWAYS', dn[always]))
    print(summarize('SOMETIMES', dn[somet]))
    print(summarize('NEVER', dn[never]))

    m_always = dn[always].mean().item() if int(always.sum()) else float('nan')
    m_never  = dn[never].mean().item()  if int(never.sum())  else float('nan')
    ratio    = m_never / (m_always + 1e-12)
    print(f'\n  RATIO  mean||delta||_NEVER / mean||delta||_ALWAYS = {ratio:.4f}')
    if ratio > 0.5:
        print('  -> WRAPPING CONFIRMED. Unsupervised geometry receives an edit of')
        print('     comparable strength to supervised geometry. The LoRA is')
        print('     generalising the appearance around the object by feature')
        print('     similarity, because it has no positional input.')
    elif ratio < 0.1:
        print('  -> NOT wrapping. The edit is confined to what the camera saw;')
        print('     the artifact is a front decal, not a wrap.')
    else:
        print('  -> PARTIAL wrapping.')

    # ── 2. does feature similarity drive the leaked delta? ───────────────────
    print('\n' + '=' * 74)
    print('[2] FEATURE-SIMILARITY COUPLING (never-visible voxels)')
    print('=' * 74)
    g = torch.Generator(device='cpu').manual_seed(0)
    nidx = torch.nonzero(never, as_tuple=False).squeeze(1)
    aidx = torch.nonzero(always, as_tuple=False).squeeze(1)
    r, corr, sims_np, dn_np = float('nan'), None, None, None

    if nidx.numel() > 10 and aidx.numel() > 10:
        ns = nidx[torch.randperm(nidx.numel(), generator=g)[:args.n_sample].to(DEVICE)]
        rs = aidx[torch.randperm(aidx.numel(), generator=g)[:args.n_ref].to(DEVICE)]
        xn = torch.nn.functional.normalize(x[ns], dim=1)
        xr = torch.nn.functional.normalize(x[rs], dim=1)
        best = torch.empty(xn.shape[0], device=DEVICE)
        step = 512
        for i in range(0, xn.shape[0], step):
            best[i:i + step] = (xn[i:i + step] @ xr.T).max(dim=1).values
        dsel = dn[ns]
        sims_np, dn_np = best.cpu().numpy(), dsel.cpu().numpy()
        bm, dm = best - best.mean(), dsel - dsel.mean()
        denom = (bm.norm() * dm.norm()).item()
        r = (bm @ dm).item() / denom if denom > 1e-12 else float('nan')
        print(f'  sampled {xn.shape[0]:,} NEVER voxels vs {xr.shape[0]:,} ALWAYS voxels')
        print(f'  max cosine similarity to a seen voxel: '
              f'mean={best.mean():.4f}  median={best.median():.4f}  min={best.min():.4f}')
        print(f'\n  Pearson r( max_cos_sim , ||delta|| ) = {r:+.4f}')
        if r > 0.2:
            print('  -> MECHANISM CONFIRMED: unseen voxels that resemble seen voxels')
            print('     inherit a larger appearance edit. The wrap is driven by')
            print('     feature similarity, not by chance.')
        elif r < -0.2:
            print('  -> Inverse coupling: the largest deltas land on voxels LEAST')
            print('     like anything seen — that is extrapolation blow-up, not a wrap.')
        else:
            print('  -> No clear coupling; the leaked delta is roughly uniform.')

    # ── 3. how much energy does the mask remove? ─────────────────────────────
    print('\n' + '=' * 74)
    print('[3] WHAT VISIBILITY MASKING REMOVES')
    print('=' * 74)
    e_tot = (dn ** 2).sum().item()
    e_nev = (dn[never] ** 2).sum().item() if int(never.sum()) else 0.0
    e_som = (dn[somet] ** 2).sum().item() if int(somet.sum()) else 0.0
    print(f'  total delta energy            : {e_tot:.4e}')
    print(f'  on NEVER-visible voxels       : {e_nev:.4e}  ({100*e_nev/(e_tot+1e-12):.2f}%)')
    print(f'  on SOMETIMES-visible voxels   : {e_som:.4e}  ({100*e_som/(e_tot+1e-12):.2f}%)')
    print(f'\n  A hard visibility mask deletes {100*e_nev/(e_tot+1e-12):.2f}% of the applied')
    print('  edit, all of it on geometry the training camera never observed.')

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.suptitle(f'Texture wrapping — {RUN_DIR.name}  frame {args.frame}', fontsize=11)

    for m, lbl, col in ((always, 'always visible', '#98c379'),
                        (somet, 'sometimes', '#d19a66'),
                        (never, 'never visible', '#e06c75')):
        v = dn[m].cpu().numpy()
        if v.size:
            axes[0].hist(v, bins=80, alpha=0.55, label=f'{lbl} (n={v.size:,})',
                         color=col, density=True)
    axes[0].set_xlabel('||delta|| per voxel'); axes[0].set_ylabel('density')
    axes[0].set_title(f'Edit magnitude by visibility  (NEVER/ALWAYS = {ratio:.2f})')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    if sims_np is not None:
        axes[1].scatter(sims_np, dn_np, s=3, alpha=0.25, color='#61afef')
        axes[1].set_xlabel('max cosine similarity to a front-visible voxel')
        axes[1].set_ylabel('||delta||')
        axes[1].set_title(f'Never-seen voxels: similarity vs edit  (r = {r:+.3f})')
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'texture_wrapping.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'\n[SAVE] {OUT_DIR / "texture_wrapping.png"}')

    json.dump({
        'run_dir': str(RUN_DIR), 'ckpt': str(CKPT), 'frame': args.frame,
        'delta_channels': args.delta_channels,
        'n_always': int(always.sum()), 'n_sometimes': int(somet.sum()),
        'n_never': int(never.sum()), 'n_total': int(dn.numel()),
        'mean_delta_always': m_always, 'mean_delta_never': m_never,
        'ratio_never_over_always': ratio,
        'pearson_sim_vs_delta': r,
        'energy_frac_never': e_nev / (e_tot + 1e-12),
        'energy_frac_sometimes': e_som / (e_tot + 1e-12),
    }, open(OUT_DIR / 'wrapping.json', 'w'), indent=2)
    print(f'[SAVE] {OUT_DIR / "wrapping.json"}')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
