"""
Step 5b — Build alignment matrices for ALL Phase C and D configs.

For each config, computes the smoothed DINOv2 token for frame 75 using that
config's exact blending formula, then runs ONE forward pass at t=500 through
the flow model to extract cross-attention and build voxel_to_token.pt.

Artifacts written to:
  artifacts/phase_c/v1/{C0..C10}/voxel_to_token.pt  + alignment.log
  artifacts/phase_c/v2/voxel_to_token.pt             + alignment.log
  artifacts/phase_d/v1/{D0..D6}/voxel_to_token.pt   + alignment.log
  artifacts/phase_d/v2/voxel_to_token.pt             + alignment.log

The original artifacts/voxel_to_token.pt (raw frame 75) is NOT touched.
Stages 1-4 are NOT touched.

Usage:
  python step05b_build_matrices.py
  python step05b_build_matrices.py --phase c --variant v1   # subset
  python step05b_build_matrices.py --phase d --variant v2
"""

import sys, os, time, argparse as _ap
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent
_PIPE     = _HERE.parent / 'dynamic_texture_trellis_pipeline'
ARTIFACTS = _HERE / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)

_LOG_PATH = ARTIFACTS / 'build_matrices.log'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

_tee = _Tee(_LOG_PATH)
sys.stdout = _tee
sys.stderr = _tee

# ── imports after Tee ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp

GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')

PRETRAINED  = 'microsoft/TRELLIS-image-large'
DEVICE      = torch.device('cuda')
STRUCT_SEED = 42     # voxel structure — must match _base.py
ALIGN_SEED  = 81     # = base_seed(6) + frame(75), same as step05
ALIGN_T     = 500.0
FRAME       = 75

_SCALE = 1024 ** -0.5
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── Config tables (exact from phase_c/run_v1.py and phase_d/run_v1.py) ───────

PHASE_C_V1 = [
    dict(tag='C0',  lc=1.0, ln=0.0),
    dict(tag='C1',  lc=0.9, ln=0.1),
    dict(tag='C2',  lc=0.8, ln=0.2),
    dict(tag='C3',  lc=0.7, ln=0.3),
    dict(tag='C4',  lc=0.6, ln=0.4),
    dict(tag='C5',  lc=0.5, ln=0.5),
    dict(tag='C6',  lc=0.4, ln=0.6),
    dict(tag='C7',  lc=0.3, ln=0.7),
    dict(tag='C8',  lc=0.2, ln=0.8),
    dict(tag='C9',  lc=0.1, ln=0.9),
    dict(tag='C10', lc=0.0, ln=1.0),
]

PHASE_D_V1 = [
    dict(tag='D0', lp=0.00, lc=1.00, ln=0.00),
    dict(tag='D1', lp=0.10, lc=0.80, ln=0.10),
    dict(tag='D2', lp=0.20, lc=0.60, ln=0.20),
    dict(tag='D3', lp=0.25, lc=0.50, ln=0.25),
    dict(tag='D4', lp=0.33, lc=0.33, ln=0.33),
    dict(tag='D5', lp=0.40, lc=0.20, ln=0.40),
    dict(tag='D6', lp=0.50, lc=0.00, ln=0.50),
]


# ── Smoothing formulas (exact match to phase C/D) ────────────────────────────

def blend_c_v1(tok_curr, tok_next, lc, ln):
    return lc * tok_curr + ln * tok_next                    # (1374, 1024)


def blend_c_v2(tok_curr, tok_next):
    Q       = tok_curr.unsqueeze(1)                         # (1374, 1, 1024)
    K       = torch.stack([tok_curr, tok_next], dim=1)      # (1374, 2, 1024)
    scores  = torch.bmm(Q, K.transpose(1, 2)) * _SCALE     # (1374, 1, 2)
    attn    = torch.softmax(scores, dim=-1)                  # (1374, 1, 2)
    blended = torch.bmm(attn, K).squeeze(1)                 # (1374, 1024)
    attn_w  = attn.squeeze(1)                               # (1374, 2)
    return blended, attn_w


def blend_d_v1(tok_prev, tok_curr, tok_next, lp, lc, ln):
    return lp * tok_prev + lc * tok_curr + ln * tok_next    # (1374, 1024)


def blend_d_v2(tok_prev, tok_curr, tok_next):
    Q       = tok_curr.unsqueeze(1)                                    # (1374, 1, 1024)
    K       = torch.stack([tok_prev, tok_curr, tok_next], dim=1)      # (1374, 3, 1024)
    scores  = torch.bmm(Q, K.transpose(1, 2)) * _SCALE                # (1374, 1, 3)
    attn    = torch.softmax(scores, dim=-1)                             # (1374, 1, 3)
    blended = torch.bmm(attn, K).squeeze(1)                            # (1374, 1024)
    attn_w  = attn.squeeze(1)                                          # (1374, 3)
    return blended, attn_w


# ── DINOv2 encoder ────────────────────────────────────────────────────────────

def encode_frame(dino_model, frame_idx):
    img_path = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    img  = Image.open(img_path).convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = dino_model(x, is_training=True)['x_prenorm']   # (1, 1374, 1024)
        tokens = F.layer_norm(feats, feats.shape[-1:])           # (1, 1374, 1024)
    return tokens.squeeze(0)                                      # (1374, 1024)


# ── Alignment: hook all 24 blocks, run ONE forward, return voxel_to_token ─────

def run_alignment(flow_model, noise_sp, cond_gl, config_label):
    """
    Hooks all 24 cross_attn blocks, runs ONE forward at t=ALIGN_T,
    averages attention across blocks and heads, returns voxel_to_token (NVP,).
    """
    accum      = [None]
    nv_patched = [None]
    hooks_fired = [0]

    def make_hook(blk_idx):
        def hook(module, args, output):
            x       = args[0]
            context = args[1]
            nv      = x.feats.shape[0]
            nh      = module.num_heads
            hd      = module.channels // nh
            sc      = hd ** -0.5
            w_dtype = module.to_q.weight.dtype

            q       = module.to_q(x.feats.to(w_dtype)).reshape(nv, nh, hd).float()
            kv_proj = module.to_kv(context[0].to(w_dtype))
            kv      = kv_proj.reshape(-1, 2, nh, hd)
            k       = kv[:, 0].float()

            scores = torch.einsum('nhd,mhd->nhm', q, k) * sc   # (nv, nh, 1374)
            attn   = torch.softmax(scores, dim=-1)               # (nv, nh, 1374)

            if accum[0] is None:
                nv_patched[0] = nv
                accum[0] = torch.zeros(nv, 1374, dtype=torch.float32)

            accum[0].add_(attn.mean(dim=1).cpu())
            hooks_fired[0] += 1

            if blk_idx == 0:
                row_sum = attn.mean(dim=1).sum(dim=1)
                print(f'    [HOOK blk={blk_idx:02d}] nv={nv}  '
                      f'row_sum mean={row_sum.mean():.4f}  '
                      f'patch_max={attn.mean(1)[:, 5:].max():.4f}')
            elif blk_idx == 23:
                row_sum = attn.mean(dim=1).sum(dim=1)
                print(f'    [HOOK blk={blk_idx:02d}] nv={nv}  '
                      f'row_sum mean={row_sum.mean():.4f}  '
                      f'patch_max={attn.mean(1)[:, 5:].max():.4f}')
        return hook

    handles = []
    for blk_idx, block in enumerate(flow_model.blocks):
        h = block.cross_attn.register_forward_hook(make_hook(blk_idx))
        handles.append(h)
    print(f'  [{config_label}] registered {len(handles)} hooks')

    t_ten = torch.tensor([ALIGN_T], device=DEVICE, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        _ = flow_model(noise_sp, t_ten, cond_gl)
    print(f'  [{config_label}] forward done in {time.time()-t0:.2f}s  hooks_fired={hooks_fired[0]}')

    for h in handles:
        h.remove()

    assert hooks_fired[0] == 24, f'Expected 24 hooks, got {hooks_fired[0]}'

    NVP       = nv_patched[0]
    attn_mean = accum[0] / 24.0                              # (NVP, 1374)
    row_sum   = attn_mean.sum(dim=1)
    print(f'  [{config_label}] attn_mean shape={tuple(attn_mean.shape)}  '
          f'row_sum: mean={row_sum.mean():.5f} min={row_sum.min():.5f} max={row_sum.max():.5f}')
    print(f'  [{config_label}] col 0 (CLS)   mean={attn_mean[:,0].mean():.6f}')
    print(f'  [{config_label}] col 1-4 (REG) mean={attn_mean[:,1:5].mean():.6f}')
    print(f'  [{config_label}] col 5-1373    mean={attn_mean[:,5:].mean():.6f}  '
          f'max={attn_mean[:,5:].max():.6f}')

    voxel_to_token = attn_mean[:, 5:].argmax(dim=1) + 5     # (NVP,) int64
    print(f'  [{config_label}] voxel_to_token shape={tuple(voxel_to_token.shape)}  '
          f'range=[{voxel_to_token.min()}, {voxel_to_token.max()}]  '
          f'unique={voxel_to_token.unique().numel()}')

    return voxel_to_token, NVP


# ── Save artifact + per-config log ───────────────────────────────────────────

def save_artifact(voxel_to_token, out_dir, config_label, blending_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / 'voxel_to_token.pt'
    log_path = out_dir / 'alignment.log'

    torch.save(voxel_to_token, art_path)

    # verification gates
    loaded = torch.load(art_path, weights_only=True)
    gates = {
        'shape=(NVP,) int64':      (tuple(loaded.shape) == (voxel_to_token.shape[0],) and loaded.dtype == torch.int64),
        'values in [5,1373]':      (int(loaded.min()) >= 5 and int(loaded.max()) <= 1373),
        'unique >= 10':            (loaded.unique().numel() >= 10),
        'round-trip matches':      (loaded == voxel_to_token).all().item(),
    }
    all_pass = all(gates.values())

    with open(log_path, 'w') as lf:
        lf.write(f'config     : {config_label}\n')
        lf.write(f'blending   : {blending_info}\n')
        lf.write(f'frame      : {FRAME}\n')
        lf.write(f'ALIGN_T    : {ALIGN_T}\n')
        lf.write(f'ALIGN_SEED : {ALIGN_SEED}\n')
        lf.write(f'STRUCT_SEED: {STRUCT_SEED}\n')
        lf.write(f'artifact   : {art_path}\n')
        lf.write(f'shape      : {tuple(voxel_to_token.shape)}\n')
        lf.write(f'range      : [{int(voxel_to_token.min())}, {int(voxel_to_token.max())}]\n')
        lf.write(f'unique     : {voxel_to_token.unique().numel()}\n')
        for name, passed in gates.items():
            lf.write(f'  [{"PASS" if passed else "FAIL"}] {name}\n')
        lf.write(f'ALL GATES {"PASSED" if all_pass else "FAILED"}\n')

    print(f'  [{config_label}] saved {art_path}  ({art_path.stat().st_size} bytes)')
    print(f'  [{config_label}] log   {log_path}')
    for name, passed in gates.items():
        print(f'    [{"PASS" if passed else "FAIL"}] {name}')
    if all_pass:
        print(f'  [{config_label}] ALL GATES PASSED')
    else:
        print(f'  [{config_label}] *** GATE FAILURE — CHECK LOG ***')

    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--phase',   type=str, default='all',
                        choices=['all', 'c', 'd'],
                        help='which phase to build: all, c, d')
    parser.add_argument('--variant', type=str, default='all',
                        choices=['all', 'v1', 'v2'],
                        help='which variant: all, v1, v2')
    args = parser.parse_args()

    do_c_v1 = args.phase in ('all', 'c') and args.variant in ('all', 'v1')
    do_c_v2 = args.phase in ('all', 'c') and args.variant in ('all', 'v2')
    do_d_v1 = args.phase in ('all', 'd') and args.variant in ('all', 'v1')
    do_d_v2 = args.phase in ('all', 'd') and args.variant in ('all', 'v2')

    n_matrices = (11 if do_c_v1 else 0) + (1 if do_c_v2 else 0) + \
                 (7  if do_d_v1 else 0) + (1 if do_d_v2 else 0)

    print('=' * 72)
    print('Step 5b — Build alignment matrices for Phase C and D configs')
    print('=' * 72)
    print(f'  phase filter   : {args.phase}')
    print(f'  variant filter : {args.variant}')
    print(f'  matrices to build: {n_matrices}')
    print(f'    phase_c v1 : {"YES (C0-C10, 11 configs)" if do_c_v1 else "skip"}')
    print(f'    phase_c v2 : {"YES (1 config)"           if do_c_v2 else "skip"}')
    print(f'    phase_d v1 : {"YES (D0-D6, 7 configs)"  if do_d_v1 else "skip"}')
    print(f'    phase_d v2 : {"YES (1 config)"           if do_d_v2 else "skip"}')
    print(f'  FRAME        : {FRAME}  (74=prev, 75=curr, 76=next)')
    print(f'  ALIGN_T      : {ALIGN_T}')
    print(f'  ALIGN_SEED   : {ALIGN_SEED}')
    print(f'  STRUCT_SEED  : {STRUCT_SEED}')
    print(f'  original artifact (untouched): {ARTIFACTS}/voxel_to_token.pt')
    print(f'  log          : {_LOG_PATH}')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\n[STEP 1] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    print(f'  flow_model type     : {type(flow_model).__name__}')
    print(f'  num_blocks          : {flow_model.num_blocks}')
    print(f'  num_heads           : {flow_model.blocks[0].cross_attn.num_heads}')
    print(f'  channels            : {flow_model.blocks[0].cross_attn.channels}')
    print(f'  qk_rms_norm_cross   : {flow_model.blocks[0].cross_attn.qk_rms_norm}  (expected False)')

    # ── Sample voxel structure ────────────────────────────────────────────────
    print(f'\n[STEP 2] Sampling voxel structure (STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(GT_FRAMES_DIR / f'frame_{FRAME:04d}.png').convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  coords shape : {tuple(coords.shape)}  N_vox={N_vox}')
    print(f'  voxel x range: [{coords[:,1].min()}, {coords[:,1].max()}]')
    print(f'  voxel y range: [{coords[:,2].min()}, {coords[:,2].max()}]')
    print(f'  voxel z range: [{coords[:,3].min()}, {coords[:,3].max()}]')

    # ── Encode frames 74, 75, 76 ──────────────────────────────────────────────
    print(f'\n[STEP 3] Encoding frames 74, 75, 76 with DINOv2...')
    dino = pipeline.models['image_cond_model'].to(DEVICE)

    raw_tokens = {}
    for fi in [74, 75, 76]:
        tok = encode_frame(dino, fi)
        raw_tokens[fi] = tok
        print(f'  frame {fi}  shape={tuple(tok.shape)}  '
              f'mean={tok.float().mean():.5f}  std={tok.float().std():.5f}  '
              f'first5={tok[5, :5].tolist()}')

    dino.cpu()
    torch.cuda.empty_cache()
    print('  DINOv2 offloaded to CPU')

    # Offload everything except flow model
    keep = {'slat_flow_model'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()
    print('  Non-flow models offloaded')

    flow_model.eval()

    # ── Build noise SparseTensor (ALIGN_SEED, fixed) ──────────────────────────
    print(f'\n[STEP 4] Building noise SparseTensor (ALIGN_SEED={ALIGN_SEED})...')
    torch.manual_seed(ALIGN_SEED)
    noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    noise_sp    = sp.SparseTensor(feats=noise_feats, coords=coords)
    print(f'  noise_feats shape={tuple(noise_feats.shape)}  '
          f'mean={noise_feats.mean():.6f}  std={noise_feats.std():.6f}')
    print(f'  first 5 values: {noise_feats[0, :5].tolist()}')

    # ── Run alignment for each config ─────────────────────────────────────────
    results_summary = []
    total_t0 = time.time()

    tok74 = raw_tokens[74]
    tok75 = raw_tokens[75]
    tok76 = raw_tokens[76]

    # ── Phase C v1 ────────────────────────────────────────────────────────────
    if do_c_v1:
        print(f'\n{"="*72}')
        print('PHASE C  v1  —  K = lc*tok[75] + ln*tok[76]')
        print(f'{"="*72}')
        for cfg in PHASE_C_V1:
            tag = cfg['tag']
            lc, ln = cfg['lc'], cfg['ln']
            print(f'\n--- {tag}  lc={lc}  ln={ln} ---')

            K_hat = blend_c_v1(tok75, tok76, lc, ln)
            print(f'  K_hat shape={tuple(K_hat.shape)}  '
                  f'mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
            diff = (K_hat - tok75).float()
            print(f'  K_hat vs raw tok75: diff_mean={diff.mean():.5f}  diff_max={diff.abs().max():.5f}')

            cond_gl = K_hat.unsqueeze(0)   # (1, 1374, 1024)

            # reset accum by rebuilding noise with same seed
            torch.manual_seed(ALIGN_SEED)
            noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
            noise_sp    = sp.SparseTensor(feats=noise_feats, coords=coords)

            vox_to_tok, NVP = run_alignment(flow_model, noise_sp, cond_gl, tag)

            out_dir  = ARTIFACTS / 'phase_c' / 'v1' / tag
            all_pass = save_artifact(vox_to_tok, out_dir, tag,
                                     f'lc={lc}*tok[75] + ln={ln}*tok[76]')
            results_summary.append((f'phase_c/v1/{tag}', all_pass))
            del K_hat, cond_gl, noise_sp, noise_feats

    # ── Phase C v2 ────────────────────────────────────────────────────────────
    if do_c_v2:
        print(f'\n{"="*72}')
        print('PHASE C  v2  —  attention blend Q=tok[75], K=[tok[75], tok[76]]')
        print(f'{"="*72}')

        K_hat, attn_w = blend_c_v2(tok75, tok76)
        attn_mean_cv2 = attn_w.mean(dim=0)   # (2,)
        print(f'  K_hat shape={tuple(K_hat.shape)}  '
              f'mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
        print(f'  attn weights mean: curr={attn_mean_cv2[0]:.4f}  next={attn_mean_cv2[1]:.4f}  '
              f'sum={attn_mean_cv2.sum():.4f}')
        print(f'  curr>next at {(attn_w[:,0]>attn_w[:,1]).float().mean()*100:.1f}% of 1374 positions')
        diff = (K_hat - tok75).float()
        print(f'  K_hat vs raw tok75: diff_mean={diff.mean():.5f}  diff_max={diff.abs().max():.5f}')

        cond_gl = K_hat.unsqueeze(0)

        torch.manual_seed(ALIGN_SEED)
        noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
        noise_sp    = sp.SparseTensor(feats=noise_feats, coords=coords)

        vox_to_tok, NVP = run_alignment(flow_model, noise_sp, cond_gl, 'C_v2')

        out_dir  = ARTIFACTS / 'phase_c' / 'v2'
        all_pass = save_artifact(vox_to_tok, out_dir, 'C_v2',
                                 f'attn blend: Q=tok[75], K=[tok[75],tok[76]], '
                                 f'mean_w=[{attn_mean_cv2[0]:.4f},{attn_mean_cv2[1]:.4f}]')
        results_summary.append(('phase_c/v2', all_pass))
        del K_hat, cond_gl, noise_sp, noise_feats

    # ── Phase D v1 ────────────────────────────────────────────────────────────
    if do_d_v1:
        print(f'\n{"="*72}')
        print('PHASE D  v1  —  K = lp*tok[74] + lc*tok[75] + ln*tok[76]')
        print(f'{"="*72}')
        for cfg in PHASE_D_V1:
            tag = cfg['tag']
            lp, lc, ln = cfg['lp'], cfg['lc'], cfg['ln']
            print(f'\n--- {tag}  lp={lp}  lc={lc}  ln={ln} ---')

            K_hat = blend_d_v1(tok74, tok75, tok76, lp, lc, ln)
            print(f'  K_hat shape={tuple(K_hat.shape)}  '
                  f'mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
            diff = (K_hat - tok75).float()
            print(f'  K_hat vs raw tok75: diff_mean={diff.mean():.5f}  diff_max={diff.abs().max():.5f}')

            cond_gl = K_hat.unsqueeze(0)

            torch.manual_seed(ALIGN_SEED)
            noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
            noise_sp    = sp.SparseTensor(feats=noise_feats, coords=coords)

            vox_to_tok, NVP = run_alignment(flow_model, noise_sp, cond_gl, tag)

            out_dir  = ARTIFACTS / 'phase_d' / 'v1' / tag
            all_pass = save_artifact(vox_to_tok, out_dir, tag,
                                     f'lp={lp}*tok[74] + lc={lc}*tok[75] + ln={ln}*tok[76]')
            results_summary.append((f'phase_d/v1/{tag}', all_pass))
            del K_hat, cond_gl, noise_sp, noise_feats

    # ── Phase D v2 ────────────────────────────────────────────────────────────
    if do_d_v2:
        print(f'\n{"="*72}')
        print('PHASE D  v2  —  attention blend Q=tok[75], K=[tok[74], tok[75], tok[76]]')
        print(f'{"="*72}')

        K_hat, attn_w = blend_d_v2(tok74, tok75, tok76)
        attn_mean_dv2 = attn_w.mean(dim=0)   # (3,)
        print(f'  K_hat shape={tuple(K_hat.shape)}  '
              f'mean={K_hat.float().mean():.5f}  std={K_hat.float().std():.5f}')
        print(f'  attn weights mean: prev={attn_mean_dv2[0]:.4f}  curr={attn_mean_dv2[1]:.4f}  '
              f'next={attn_mean_dv2[2]:.4f}  sum={attn_mean_dv2.sum():.4f}')
        print(f'  curr>both others at '
              f'{((attn_w[:,1]>attn_w[:,0]) & (attn_w[:,1]>attn_w[:,2])).float().mean()*100:.1f}% of positions')
        diff = (K_hat - tok75).float()
        print(f'  K_hat vs raw tok75: diff_mean={diff.mean():.5f}  diff_max={diff.abs().max():.5f}')

        cond_gl = K_hat.unsqueeze(0)

        torch.manual_seed(ALIGN_SEED)
        noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
        noise_sp    = sp.SparseTensor(feats=noise_feats, coords=coords)

        vox_to_tok, NVP = run_alignment(flow_model, noise_sp, cond_gl, 'D_v2')

        out_dir  = ARTIFACTS / 'phase_d' / 'v2'
        all_pass = save_artifact(vox_to_tok, out_dir, 'D_v2',
                                 f'attn blend: Q=tok[75], K=[tok[74],tok[75],tok[76]], '
                                 f'mean_w=[{attn_mean_dv2[0]:.4f},{attn_mean_dv2[1]:.4f},{attn_mean_dv2[2]:.4f}]')
        results_summary.append(('phase_d/v2', all_pass))
        del K_hat, cond_gl, noise_sp, noise_feats

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f'\n{"="*72}')
    print(f'FINAL SUMMARY  —  {n_matrices} matrices  total={total_elapsed:.1f}s')
    print(f'{"="*72}')
    all_passed = True
    for label, passed in results_summary:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {label}')
        if not passed:
            all_passed = False

    print(f'\n  original artifact (untouched): {ARTIFACTS}/voxel_to_token.pt')
    print(f'  log: {_LOG_PATH}')
    if all_passed:
        print(f'\n  ALL {len(results_summary)} MATRICES PASSED GATES')
    else:
        print(f'\n  *** SOME MATRICES FAILED — CHECK LOGS ***')
    print('=' * 72)


if __name__ == '__main__':
    main()
