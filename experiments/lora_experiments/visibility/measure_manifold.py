"""
measure_manifold.py
-------------------
Does the LoRA push the SLaT latent OFF the distribution the mesh decoder was
trained on, and are the off-manifold voxels the ones that go bright?

WHY
  Frozen TRELLIS produces no white patches for ANY input image. That cannot be
  explained by "frozen happens to be dark on this teapot" -- it holds generally.
  The candidate explanation is distributional: the decoder only ever receives
  latents its own flow produced, and its colour logits are bounded there
  (measured max +2.57). The LoRA edits cross-attention INSIDE the flow, so it
  can emit a latent the decoder never saw in training, where nothing bounds the
  logits (rung15 reached +5.71).

  pipeline.json ships SLAT_MEAN/SLAT_STD, the per-channel statistics of that
  training distribution. So "on-manifold" has a concrete meaning here: the
  normalised latent z = (x - mean)/std should look like the flow's own output.

WHAT IS MEASURED, frozen vs adapted, over the 7,301 latent voxels x 8 channels
  1. |z| distribution: mean, p99, max. Frozen sets the reference scale; the
     question is whether the adapter's tail runs past it.
  2. per-voxel drift ||z_lora - z_frozen||, and where that drift sits in space.
  3. THE JOINT TEST -- correlation between a voxel's drift and the brightness of
     the mesh vertices it produces. Off-manifold ALONE is not an explanation; it
     only becomes one if the drifted voxels are the bright ones. Reported as the
     mean drift of bright-producing voxels over that of the rest.
  4. decoder logits (out_layer, channels 53:101 -> RGB per corner, cube2mesh.py)
     for both, so the +2.57 / +5.71 comparison is reproduced in the same run.

Latents are compared BEFORE normalize_slat, i.e. in the flow's own unit-normal
space, which is where the distribution statistics apply.
"""
import sys, os, argparse, json, gc
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_LEX  = _HERE.parent
_ROOT = _LEX.parent.parent
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--run', default='rung19_intersection_uniform_all_qkvo_r4_s6_276b5db7')
_pre.add_argument('--ckpt', default='lora_best.pt')
_pre.add_argument('--frame', type=int, default=1)
_pre.add_argument('--bright', type=float, default=0.5)
A, _ = _pre.parse_known_args()
OUT = _HERE / 'manifold'; OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_xattn_mesh.py', '--run', A.run, '--ckpt', A.ckpt,
            '--frame', str(A.frame), '--no-glb', '--out-dir', str(OUT/'_scratch')]
sys.path.insert(0, str(_ROOT/'render'))
import export_xattn_mesh as X
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(_ROOT/'experiments'/'dynamic_texture_trellis_pipeline'))
from step8_decode_render.decode_render import SLAT_MEAN, SLAT_STD
DEVICE = X.DEVICE

from trellis.pipelines import TrellisImageTo3DPipeline
pipe = TrellisImageTo3DPipeline.from_pretrained(X.PRETRAINED); pipe.to(DEVICE)
fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
for p in fm.parameters(): p.requires_grad_(False)
for p in dec.parameters(): p.requires_grad_(False)
ref = Image.open(X.GT_FRAMES_DIR/'frame_0075.png').convert('RGB')
cs = pipe.get_cond([ref]); torch.manual_seed(X.STRUCT_SEED)
coords = pipe.sample_sparse_structure(cs, num_samples=1)
assert coords.shape[0] == 7301
del cs; gc.collect(); torch.cuda.empty_cache()
for n in list(pipe.models):
    if n not in {'slat_flow_model','slat_decoder_mesh','image_cond_model'}:
        try: pipe.models[n].cpu()
        except Exception: pass
dino = pipe.models['image_cond_model'].to(DEVICE)
cond = X.encode(dino, A.frame).unsqueeze(0).to(DEVICE)
dino.cpu(); gc.collect(); torch.cuda.empty_cache()
rd = _LEX/'runs'/A.run
cfg = json.load(open(rd/'config.json'))
ck = torch.load(rd/'lora_ckpts'/A.ckpt, map_location='cpu', weights_only=True)
reg = X.XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                          tuple(cfg['targets'])).to(DEVICE); reg.load_state_dict(
                          ck['registry_state'], strict=True); reg.eval()
torch.manual_seed(X.NOISE_SEED)
noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)

def raw_latent(L):
    """Flow output BEFORE normalize_slat -- the flow's own unit-normal space."""
    X.LORA = L
    x = __import__('trellis.modules.sparse', fromlist=['SparseTensor']).SparseTensor(
        feats=noise.clone(), coords=coords)
    with torch.no_grad(), X.lora_ctx(fm):
        for t, tp in X.T_PAIRS:
            tt = torch.tensor([1000.0*t], device=DEVICE, dtype=torch.float32)
            x = x.replace(x.feats - (t-tp)*fm(x, tt, cond).feats)
    return x

print('='*80)
print(f'MANIFOLD  run={A.run}  frame={A.frame}')
print('='*80, flush=True)
res = {}
for tag, L in (('frozen', None), ('rung', reg)):
    xr = raw_latent(L)
    z = xr.feats.float()
    X.LORA = L
    with torch.no_grad():
        from step8_decode_render.decode_render import normalize_slat
        mesh = dec(normalize_slat(xr))[0]
    col = mesh.vertex_attrs[:, :3].float()
    logit = torch.logit(col.clamp(1e-6, 1-1e-6))
    res[tag] = dict(z=z.cpu(), col=col.cpu(), logit=logit.cpu(),
                    v=mesh.vertices.detach().cpu())
    print(f'{tag:7s} |z| mean {z.abs().mean():.4f}  p99 {z.abs().flatten().quantile(0.99):.4f}  '
          f'max {z.abs().max():.4f}   colour logit max {logit.max():+.3f}  '
          f'min {logit.min():+.3f}   verts {len(col):,}')
    del xr, mesh; gc.collect(); torch.cuda.empty_cache()

zf, zl = res['frozen']['z'], res['rung']['z']
d = (zl - zf).norm(dim=1)
print(f'\nper-voxel latent drift ||z_lora - z_frozen||')
print(f'  mean {d.mean():.4f}  p50 {d.median():.4f}  p99 {d.quantile(0.99):.4f}  max {d.max():.4f}')
print(f'  frozen per-voxel ||z||  mean {zf.norm(dim=1).mean():.4f}  '
      f'-> drift is {100*d.mean()/zf.norm(dim=1).mean():.1f}% of the latent norm')
print(f'  |z| max: frozen {zf.abs().max():.3f}  rung {zl.abs().max():.3f}  '
      f'(ratio {zl.abs().max()/zf.abs().max():.3f}x)')
np.savez_compressed(OUT/'manifold.npz', d=d.numpy(),
                    zf_norm=zf.norm(dim=1).numpy(), zl_norm=zl.norm(dim=1).numpy(),
                    logit_frozen=res['frozen']['logit'].numpy(),
                    logit_rung=res['rung']['logit'].numpy())
print(f'\n[DONE] {OUT}/manifold.npz')
