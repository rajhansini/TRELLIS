"""
Stage 2.5 — 3D Alignment (offline, one-time).

Hooks all 24 cross-attn blocks in the SLAT flow model, runs ONE forward pass
at t=500, and computes the mean attention map (softmax applied PER block
BEFORE averaging across blocks and heads).

Outputs
-------
  artifacts/voxel_to_token.pt  — int64 (7301,), values in [5, 1373]
  artifacts/alignment.log      — full wire log

Stages 1-4 (blending_no_lora_no_enhancement) are NOT touched.

Usage
-----
  cd /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement
  python step05_alignment.py
"""

# ── Must be first: Tee before ANY import ──────────────────────────────────────
import sys, os, time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

_HERE     = Path(__file__).resolve().parent          # experiments/enhancement
_ROOT     = _HERE.parent.parent                      # TRELLIS root
_PIPE     = _HERE.parent / 'dynamic_texture_trellis_pipeline'
ARTIFACTS = _HERE / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)

    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()


_tee = _Tee(ARTIFACTS / 'alignment.log')
sys.stdout = _tee
sys.stderr = _tee

# ── Now safe to import (prints captured by Tee) ───────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from PIL import Image

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp

# ── Constants — voxel coords seed MUST match _base.py exactly ─────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
NOISE_SEED  = 42      # voxel structure seed — same as blending_no_lora_no_enhancement/_base.py
ALIGN_SEED  = 81      # = 6 + 75, noise for the one alignment forward pass
ALIGN_T     = 500.0   # mid-noise timestep
FRAME_75    = 75
GT_FRAME_75 = str(_PIPE / '..' / '..' / '..' /
                  'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')


def main():
    print('=' * 72)
    print('Stage 2.5 — 3D Alignment (offline, one-time)')
    print('=' * 72)
    print(f'  NOISE_SEED (voxel coords) : {NOISE_SEED}  (must match _base.py)')
    print(f'  ALIGN_SEED (noise)        : {ALIGN_SEED}  (= 6 + 75)')
    print(f'  ALIGN_T                   : {ALIGN_T}')
    print(f'  FRAME                     : {FRAME_75}')
    print(f'  artifact                  : {ARTIFACTS}/voxel_to_token.pt')
    print(f'  log                       : {ARTIFACTS}/alignment.log')

    # ── Step 1: Load pipeline ──────────────────────────────────────────────────
    print('\n[STEP 1] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    ca0      = flow_model.blocks[0].cross_attn
    NUM_BLKS = flow_model.num_blocks           # 24
    IN_CH    = flow_model.in_channels          # 8
    NH       = ca0.num_heads                   # 16
    HD       = ca0.channels // NH              # 64
    SCALE    = HD ** -0.5                      # 0.125
    QK_NORM  = ca0.qk_rms_norm                # False for cross_attn (no qk_rms_norm_cross in config)

    print(f'  type            : {type(flow_model).__name__}')
    print(f'  num_blocks      : {NUM_BLKS}')
    print(f'  in_channels     : {IN_CH}')
    print(f'  num_heads       : {NH}  head_dim={HD}  scale={SCALE:.6f}')
    print(f'  to_q weight shape : {tuple(ca0.to_q.weight.shape)}  dtype={ca0.to_q.weight.dtype}')
    print(f'  to_kv weight shape: {tuple(ca0.to_kv.weight.shape)}')
    print(f'  qk_rms_norm_cross : {QK_NORM}  (expected False)')

    # ── Step 2: Sample voxel coords — NOISE_SEED=42 matches _base.py ─────────
    # NOTE: pipeline.to(DEVICE) moved image_cond_model (DINOv2) to CUDA.
    # We encode frame 75 NOW, before any offloading, to avoid DINOv2 landing on CPU.
    print(f'\n[STEP 2] Sampling voxel structure from frame 75 (NOISE_SEED={NOISE_SEED})...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]

    print(f'  coords shape : {tuple(coords.shape)}  dtype={coords.dtype}')
    print(f'  N_vox        : {N_vox}  (expected 7301)')
    print(f'  voxel x range: [{int(coords[:,1].min())}, {int(coords[:,1].max())}]')
    print(f'  voxel y range: [{int(coords[:,2].min())}, {int(coords[:,2].max())}]')
    print(f'  voxel z range: [{int(coords[:,3].min())}, {int(coords[:,3].max())}]')

    # ── Step 3: DINOv2 encode frame 75 (pipeline's DINOv2 on CUDA) ────────────
    # Use pipeline.encode_image() — same DINOv2 that _base.py experiments use.
    # Must happen BEFORE offloading (which would move image_cond_model to CPU).
    print(f'\n[STEP 3] DINOv2 encoding of frame {FRAME_75} via pipeline.encode_image()...')
    img_f75 = Image.open(GT_FRAME_75).convert('RGB')
    cond_gl = pipeline.encode_image([img_f75])       # (1, 1374, 1024) on CUDA
    tok_75  = cond_gl.squeeze(0)                     # (1374, 1024)

    print(f'  cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}')
    print(f'  tok_75  mean={tok_75.float().mean():.6f}  std={tok_75.float().std():.6f}')
    print(f'  tok_75[0]   (CLS)   first 5: {tok_75[0, :5].tolist()}')
    print(f'  tok_75[1:5] (REG)   first dim: {tok_75[1:5, 0].tolist()}')
    print(f'  tok_75[5]   (patch0) first 5: {tok_75[5, :5].tolist()}')

    # ── Offload non-flow models now that encoding is done ─────────────────────
    print(f'\n  Offloading non-flow-model models to CPU...')
    for name in list(pipeline.models.keys()):
        if name != 'slat_flow_model':
            try:
                pipeline.models[name].cpu()
            except Exception:
                pass
    torch.cuda.empty_cache()
    print(f'  Done — image_cond_model and others offloaded')

    # ── Step 4: Build noise tensor ─────────────────────────────────────────────
    print(f'\n[STEP 4] Building noise SparseTensor (ALIGN_SEED={ALIGN_SEED})...')
    torch.manual_seed(ALIGN_SEED)
    noise_feats = torch.randn(N_vox, IN_CH, device=DEVICE)
    print(f'  noise_feats shape={tuple(noise_feats.shape)}  dtype={noise_feats.dtype}')
    print(f'  noise_feats mean={noise_feats.mean():.6f}  std={noise_feats.std():.6f}')
    noise_sp = sp.SparseTensor(feats=noise_feats, coords=coords)

    # ── Step 5: Register forward hooks on all 24 cross_attn blocks ────────────
    # NOTE: The flow model patchifies the N_vox sparse voxels into a smaller set
    # of "patch tokens" before the transformer blocks. We don't know nv_patched
    # upfront, so accum is lazily initialised on the first hook call.
    print(f'\n[STEP 5] Registering hooks on {NUM_BLKS} cross_attn blocks...')
    accum      = [None]    # lazily set to (nv_patched, 1374) on first hook call
    nv_patched = [None]    # set on first hook call
    cnt        = [0]

    def make_hook(blk_i, nh, hd, sc, qkn):
        def hook_fn(module, inp, out):
            x       = inp[0]    # SparseTensor after norm2 — nv_patched × channels
            context = inp[1]    # (1, 1374, ctx_channels)

            with torch.no_grad():
                nv = x.feats.shape[0]
                w_dtype = module.to_q.weight.dtype

                # ── Q: (nv, channels) → (nv, nh, hd) fp32 ──────────────────
                q = module.to_q(x.feats.to(w_dtype))   # (nv, channels)
                q = q.reshape(nv, nh, hd).float()       # (nv, nh, hd)

                # ── K: to_kv → split → first half is K ──────────────────────
                kv_proj = module.to_kv(context[0].to(w_dtype))  # (1374, 2*channels)
                kv      = kv_proj.reshape(-1, 2, nh, hd)         # (1374, 2, nh, hd)
                k       = kv[:, 0].float()                        # (1374, nh, hd)
                del kv_proj, kv

                # ── qk_rms_norm: False for cross_attn but handled just in case
                if qkn:
                    gq  = module.q_rms_norm.gamma.float()  # (nh, hd)
                    gk  = module.k_rms_norm.gamma.float()  # (nh, hd)
                    sq  = float(module.q_rms_norm.scale)
                    sk  = float(module.k_rms_norm.scale)
                    q   = F.normalize(q, dim=-1) * gq * sq
                    k   = F.normalize(k, dim=-1) * gk * sk

                # ── scores: (nv, nh, 1374) ───────────────────────────────────
                scores = torch.einsum('nhd,mhd->nhm', q, k) * sc   # (nv, nh, 1374)
                del q, k

                # ── softmax BEFORE averaging over heads ──────────────────────
                attn     = torch.softmax(scores, dim=-1)     # (nv, nh, 1374)
                del scores

                attn_blk = attn.mean(dim=1).cpu()            # (nv, 1374) fp32 on CPU
                del attn

                # Lazy-init accumulator with the actual patched voxel count
                if accum[0] is None:
                    nv_patched[0] = nv
                    accum[0]      = torch.zeros(nv, 1374, dtype=torch.float32)
                    print(f'  [HOOK blk={blk_i:02d}] INIT accum:'
                          f'  N_vox_original={N_vox}  N_vox_patched={nv}'
                          f'  ratio={N_vox/nv:.3f}')

                accum[0].add_(attn_blk)
                cnt[0] += 1

                # Wire log for first two and last two blocks
                if blk_i in (0, 1, 22, 23):
                    rs  = attn_blk.sum(dim=1)
                    am  = attn_blk[:, 5:]   # patch-only columns
                    print(f'  [HOOK blk={blk_i:02d}]'
                          f'  nv={nv}'
                          f'  row_sum: mean={rs.mean():.4f} min={rs.min():.4f} max={rs.max():.4f}'
                          f'  patch_max={am.max():.4f}'
                          f'  argmax_patch: vox0={int(am[0].argmax())+5}'
                          f'  vox1000={int(am[min(1000,nv-1)].argmax())+5}')
        return hook_fn

    handles = []
    for bi, block in enumerate(flow_model.blocks):
        ca = block.cross_attn
        nh = ca.num_heads
        hd = ca.channels // nh
        sc = hd ** -0.5
        h  = ca.register_forward_hook(make_hook(bi, nh, hd, sc, ca.qk_rms_norm))
        handles.append(h)

    print(f'  {len(handles)} hooks registered')

    # ── Step 6: ONE forward pass at t=ALIGN_T ─────────────────────────────────
    print(f'\n[STEP 6] ONE forward pass: flow_model(noise_sp, t={ALIGN_T}, cond_gl)...')
    t_ten = torch.tensor([ALIGN_T], device=DEVICE, dtype=torch.float32)
    print(f'  t_ten={t_ten.tolist()}  dtype={t_ten.dtype}')
    print(f'  noise_sp.feats dtype={noise_sp.feats.dtype}  shape={tuple(noise_sp.feats.shape)}')
    print(f'  cond_gl dtype={cond_gl.dtype}  shape={tuple(cond_gl.shape)}')

    flow_model.eval()
    t0 = time.time()
    with torch.no_grad():
        _ = flow_model(noise_sp, t_ten, cond_gl)
    elapsed = time.time() - t0

    for h in handles:
        h.remove()

    print(f'  Forward done in {elapsed:.2f}s')
    print(f'  hooks_fired={cnt[0]}  expected={NUM_BLKS}  OK={cnt[0]==NUM_BLKS}')

    if cnt[0] != NUM_BLKS:
        print(f'  ERROR: {NUM_BLKS - cnt[0]} hooks did NOT fire. Aborting.')
        return

    NVP = nv_patched[0]   # actual patched-voxel count (1748, not 7301)

    # ── Step 7: Average across blocks ─────────────────────────────────────────
    print('\n[STEP 7] Averaging attention across blocks...')
    attn_mean = accum[0] / NUM_BLKS      # (NVP, 1374) fp32
    rs        = attn_mean.sum(dim=1)     # (NVP,) — should be ~1.0

    print(f'  N_vox_original : {N_vox}')
    print(f'  N_vox_patched  : {NVP}  (voxels seen by transformer cross-attn)')
    print(f'  attn_mean shape={tuple(attn_mean.shape)}  dtype={attn_mean.dtype}')
    print(f'  row_sum  : mean={rs.mean():.5f}  min={rs.min():.5f}  max={rs.max():.5f}  (expected ~1.0)')
    print(f'  attn_mean val range: min={attn_mean.min():.7f}  max={attn_mean.max():.7f}')
    print(f'  col 0 (CLS)        : mean={attn_mean[:,0].mean():.6f}  max={attn_mean[:,0].max():.6f}')
    print(f'  col 1-4 (REG)      : mean={attn_mean[:,1:5].mean():.6f}')
    print(f'  col 5-1373 (patch) : mean={attn_mean[:,5:].mean():.6f}  max={attn_mean[:,5:].max():.6f}')

    # ── Step 8: Argmax over patch tokens → patch_to_token ─────────────────────
    print('\n[STEP 8] patch_to_token = argmax(attn_mean[:, 5:], dim=1) + 5...')
    patch_attn  = attn_mean[:, 5:]                               # (NVP, 1369)
    vox_to_tok  = patch_attn.argmax(dim=1).to(torch.int64) + 5  # (NVP,) in [5, 1373]

    val_min    = int(vox_to_tok.min())
    val_max    = int(vox_to_tok.max())
    n_unique   = vox_to_tok.unique().numel()

    print(f'  patch_to_token shape={tuple(vox_to_tok.shape)}  dtype={vox_to_tok.dtype}')
    print(f'  value range : [{val_min}, {val_max}]  expected [5, 1373]')
    print(f'  unique tokens used : {n_unique}  (expected hundreds+)')

    # Token grid breakdown (patch tokens form a 37×37 grid)
    grid_idx = (vox_to_tok - 5).numpy()
    rows = grid_idx // 37
    cols = grid_idx % 37
    print(f'  grid row range : [{int(rows.min())}, {int(rows.max())}]  mean={rows.mean():.1f}')
    print(f'  grid col range : [{int(cols.min())}, {int(cols.max())}]  mean={cols.mean():.1f}')
    print(f'  grid center expected near (18, 18) — teapot is front-centered')

    tok_counts = Counter(vox_to_tok.numpy().tolist())
    top10      = tok_counts.most_common(10)
    print(f'  top-10 tokens by patch count: {top10}')
    print(f'  patch_to_token[0:10]     : {vox_to_tok[:10].tolist()}')
    print(f'  patch_to_token[870:880]  : {vox_to_tok[870:880].tolist()}  (mid-range)')

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    out_path = ARTIFACTS / 'voxel_to_token.pt'
    torch.save(vox_to_tok, out_path)
    print(f'\n[SAVED] {out_path}  ({out_path.stat().st_size} bytes)')
    print(f'  NOTE: shape=({NVP},) — indexes into NVP patched voxels, NOT original {N_vox}')

    # ── Step 10: Verification gates ───────────────────────────────────────────
    print('\n[STEP 10] Verification gates...')
    loaded       = torch.load(out_path)
    g_shape      = tuple(loaded.shape) == (NVP,) and loaded.dtype == torch.int64
    g_range      = val_min >= 5 and val_max <= 1373
    g_unique     = n_unique >= 50
    g_hooks      = cnt[0] == NUM_BLKS
    g_rowsum     = abs(float(rs.mean()) - 1.0) < 0.01
    g_roundtrip  = bool((loaded == vox_to_tok).all())

    def gate(ok, label):
        print(f'  [{"PASS" if ok else "FAIL"}] {label}')
        return ok

    all_ok = True
    all_ok &= gate(g_shape,     f'shape=({NVP},) dtype=int64')
    all_ok &= gate(g_range,     f'values in [5, 1373]  got [{val_min}, {val_max}]')
    all_ok &= gate(g_unique,    f'unique tokens >= 50  got {n_unique}')
    all_ok &= gate(g_hooks,     f'all {NUM_BLKS} hooks fired  got {cnt[0]}')
    all_ok &= gate(g_rowsum,    f'attn row_sum ≈ 1.0  got {float(rs.mean()):.5f}')
    all_ok &= gate(g_roundtrip, f'round-trip torch.save/load matches')

    passed = sum([g_shape, g_range, g_unique, g_hooks, g_rowsum, g_roundtrip])
    print(f'\n  {"ALL 6 GATES PASSED" if all_ok else f"FAILED {6-passed}/6 gates — check above"}')
    print(f'\n  artifact : {out_path}')
    print(f'  log      : {ARTIFACTS}/alignment.log')
    print('=' * 72)


if __name__ == '__main__':
    main()
