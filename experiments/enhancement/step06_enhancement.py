"""
Step 6 — Enhancement (variable noise, per-frame seed).

Injects additive bias (beta) into all 24 cross-attn logits during denoising.
Now supports phase C/D smoothed tokens + matching alignment matrix.

BACKWARD COMPATIBLE: default --phase raw uses original artifacts/voxel_to_token.pt
and raw tokens — identical to the original step06 behaviour.

Usage
-----
  # original behaviour (raw tokens, raw matrix)
  python step06_enhancement.py --betas 0 1 2 3 4 6 8 16 --start_frame 1 --frames 150

  # phase C v1 config C3
  python step06_enhancement.py --phase c --variant v1 --config C3 --betas 0 1 2 3 4 6 8 16 --start_frame 1 --frames 150

  # phase C v2
  python step06_enhancement.py --phase c --variant v2 --betas 0 1 2 3 4 6 8 16 --start_frame 1 --frames 150

  # phase D v1 config D4
  python step06_enhancement.py --phase d --variant v1 --config D4 --betas 0 1 2 3 4 6 8 16 --start_frame 1 --frames 150

  # phase D v2
  python step06_enhancement.py --phase d --variant v2 --betas 0 1 2 3 4 6 8 16 --start_frame 1 --frames 150

Stages 1-4 (blending_no_lora_no_enhancement) are NOT touched.
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

# ── pre-parse all args before Tee so we can name the log ─────────────────────
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--betas',       type=float, nargs='+', default=[0.0, 2.0])
_pre.add_argument('--base_seed',   type=int,   default=6)
_pre.add_argument('--start_frame', type=int,   default=75)
_pre.add_argument('--frames',      type=int,   default=1)
_pre.add_argument('--phase',       type=str,   default='raw')
_pre.add_argument('--variant',     type=str,   default='v1')
_pre.add_argument('--config',      type=str,   default='')
_PRE, _ = _pre.parse_known_args()

# results dir and log name encode the full experiment identity
_phase_tag = _PRE.phase.lower()
if _phase_tag != 'raw':
    _cfg_tag = f'_{_PRE.config}' if _PRE.config else ''
    _exp_tag = f'phase_{_phase_tag}_{_PRE.variant}{_cfg_tag}'
else:
    _exp_tag = 'raw'

RESULTS = _HERE / f'results_{_exp_tag}'
RESULTS.mkdir(parents=True, exist_ok=True)

_LOG_NAME = (f'run_{_exp_tag}_betas{"_".join(str(b) for b in _PRE.betas)}'
             f'_seed{_PRE.base_seed}.log')


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

_tee = _Tee(RESULTS / _LOG_NAME)
sys.stdout = _tee
sys.stderr = _tee

# ── imports after Tee ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import math

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp

GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
GT_FRAME_75   = GT_FRAMES_DIR / 'frame_0075.png'

PRETRAINED = 'microsoft/TRELLIS-image-large'
DEVICE     = torch.device('cuda')
STRUCT_SEED = 42
N_FRAMES    = 150

RENDER_RES = 518
RESCALE_T  = 3.0
STEPS      = 25
_t_seq     = np.linspace(1, 0, STEPS + 1)
_t_seq     = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS    = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]

_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n,0.,0.5],[0.,_fx_n,0.5],[0.,0.,1.]],
                           dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.tensor([
    [1., 0., 0., 0.],
    [0., 0.,-1., 0.],
    [0., 1., 0., 2.],
    [0., 0., 0., 1.],
], dtype=torch.float32, device=DEVICE)

SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,  1.2039363384246826
], dtype=torch.float32)
SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338
], dtype=torch.float32)

_SCALE     = 1024 ** -0.5
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── Phase C / D config tables (exact match to phase_c/run_v1.py etc.) ─────────
PHASE_C_V1 = {
    'C0':  dict(lc=1.0, ln=0.0),  'C1':  dict(lc=0.9, ln=0.1),
    'C2':  dict(lc=0.8, ln=0.2),  'C3':  dict(lc=0.7, ln=0.3),
    'C4':  dict(lc=0.6, ln=0.4),  'C5':  dict(lc=0.5, ln=0.5),
    'C6':  dict(lc=0.4, ln=0.6),  'C7':  dict(lc=0.3, ln=0.7),
    'C8':  dict(lc=0.2, ln=0.8),  'C9':  dict(lc=0.1, ln=0.9),
    'C10': dict(lc=0.0, ln=1.0),
}
PHASE_D_V1 = {
    'D0': dict(lp=0.00, lc=1.00, ln=0.00), 'D1': dict(lp=0.10, lc=0.80, ln=0.10),
    'D2': dict(lp=0.20, lc=0.60, ln=0.20), 'D3': dict(lp=0.25, lc=0.50, ln=0.25),
    'D4': dict(lp=0.33, lc=0.33, ln=0.33), 'D5': dict(lp=0.40, lc=0.20, ln=0.40),
    'D6': dict(lp=0.50, lc=0.00, ln=0.50),
}


# ── Token smoothing (exact formulas from phase_c/d run scripts) ───────────────

def smooth_c_v1(tok_curr, tok_next, lc, ln):
    return lc * tok_curr + ln * tok_next          # (1374, 1024)

def smooth_c_v2(tok_curr, tok_next):
    Q      = tok_curr.unsqueeze(1)
    K      = torch.stack([tok_curr, tok_next], dim=1)
    scores = torch.bmm(Q, K.transpose(1, 2)) * _SCALE
    attn   = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, K).squeeze(1), attn.squeeze(1)   # (1374,1024), (1374,2)

def smooth_d_v1(tok_prev, tok_curr, tok_next, lp, lc, ln):
    return lp * tok_prev + lc * tok_curr + ln * tok_next

def smooth_d_v2(tok_prev, tok_curr, tok_next):
    Q      = tok_curr.unsqueeze(1)
    K      = torch.stack([tok_prev, tok_curr, tok_next], dim=1)
    scores = torch.bmm(Q, K.transpose(1, 2)) * _SCALE
    attn   = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, K).squeeze(1), attn.squeeze(1)   # (1374,1024), (1374,3)


# ── Artifact path resolver ────────────────────────────────────────────────────

def resolve_artifact(phase, variant, config):
    """Return path to the correct voxel_to_token.pt for this experiment."""
    if phase == 'raw':
        return ARTIFACTS / 'voxel_to_token.pt'
    if variant == 'v1':
        return ARTIFACTS / f'phase_{phase}' / 'v1' / config / 'voxel_to_token.pt'
    else:
        return ARTIFACTS / f'phase_{phase}' / 'v2' / 'voxel_to_token.pt'


# ── DINOv2 encoder ────────────────────────────────────────────────────────────

def encode_frame_raw(dino_model, frame_idx):
    img_path = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    img  = Image.open(img_path).convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(torch.from_numpy(arr).permute(2,0,1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = dino_model(x, is_training=True)['x_prenorm']
        tokens = F.layer_norm(feats, feats.shape[-1:])
    return tokens.squeeze(0)   # (1374, 1024)


# ── Enhancement: patch / unpatch cross-attn ───────────────────────────────────

def patch_flow_model(flow_model, bias_map, beta, wire_check_log=None):
    bm    = bias_map.to(DEVICE)
    saved = []
    for blk_idx, block in enumerate(flow_model.blocks):
        ca   = block.cross_attn
        nh   = ca.num_heads
        hd   = ca.channels // nh
        sc   = hd ** -0.5
        orig = ca.forward

        def make_fwd(mod, nh_, hd_, sc_, bias, bt, blk_i, wc_log):
            fired = [False]
            def biased_fwd(x_st, context=None):
                nv   = x_st.feats.shape[0]
                w_dt = mod.to_q.weight.dtype
                q       = mod.to_q(x_st.feats.to(w_dt)).reshape(nv, nh_, hd_)
                kv_proj = mod.to_kv(context[0].to(w_dt))
                kv      = kv_proj.reshape(-1, 2, nh_, hd_)
                k, v    = kv[:, 0], kv[:, 1]
                scores_unbiased = torch.einsum('nhd,mhd->nhm', q.float(), k.float()) * sc_
                scores          = scores_unbiased + bt * bias.unsqueeze(1).to(scores_unbiased.dtype)
                if blk_i == 0 and wc_log is not None and not fired[0]:
                    fired[0] = True
                    with torch.no_grad():
                        ab = torch.softmax(scores_unbiased, dim=-1).mean(1)
                        aa = torch.softmax(scores,          dim=-1).mean(1)
                        asgn = bias.argmax(dim=1)
                        wc_log.append({
                            'before': ab[torch.arange(nv), asgn].mean().item(),
                            'after':  aa[torch.arange(nv), asgn].mean().item(),
                            'delta':  (aa - ab)[torch.arange(nv), asgn].mean().item(),
                            'rest_before': ((ab.sum(1) - ab[torch.arange(nv),asgn]) / 1373).mean().item(),
                            'rest_after':  ((aa.sum(1) - aa[torch.arange(nv),asgn]) / 1373).mean().item(),
                        })
                attn = torch.softmax(scores, dim=-1).to(w_dt)
                out  = torch.einsum('nhm,mhd->nhd', attn, v).reshape(nv, nh_ * hd_)
                return x_st.replace(mod.to_out(out))
            return biased_fwd

        ca.forward = make_fwd(ca, nh, hd, sc, bm, beta, blk_idx, wire_check_log)
        saved.append((ca, orig))

    print(f'  [PATCH] beta={beta}  patched {len(saved)} cross_attn blocks')
    print(f'  [PATCH] bias_map device={bm.device}  shape={tuple(bm.shape)}  nonzero={int((bm>0).sum())}')
    return saved


def unpatch_flow_model(saved):
    for ca, orig in saved:
        ca.forward = orig
    print(f'  [UNPATCH] restored {len(saved)} cross_attn blocks')


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def load_pipeline_and_coords():
    print('[LOAD] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    print(f'[LOAD] Sampling voxel structure (STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox={N_vox}  coords.shape={tuple(coords.shape)}')

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()
    return pipeline, flow_model, coords, N_vox


def denoise(flow_model, noise_sp, cond_gl):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def normalize_slat(x0):
    return x0.replace(x0.feats * SLAT_STD.to(x0.feats.device) + SLAT_MEAN.to(x0.feats.device))


def render(pipeline, slat):
    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )
    with torch.no_grad():
        mesh = pipeline.decode_slat(slat, ['mesh'])['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color', 'mask'])
    mask   = result['mask'].unsqueeze(0)
    color  = result['color'] * mask + (1.0 - mask)
    return (color.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)


def assemble_video(out_dir, start_frame):
    import subprocess
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', '25',
        '-start_number', str(start_frame),
        '-i', str(out_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(out_dir / 'render.mp4')
    ], check=True, capture_output=True)
    print(f'  video: {out_dir}/render.mp4')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--betas',       type=float, nargs='+', default=[0.0, 2.0])
    parser.add_argument('--start_frame', type=int,   default=75)
    parser.add_argument('--frames',      type=int,   default=1)
    parser.add_argument('--base_seed',   type=int,   default=6)
    parser.add_argument('--phase',       type=str,   default='raw',
                        help='raw | c | d  (default: raw = backward compatible)')
    parser.add_argument('--variant',     type=str,   default='v1',
                        help='v1 | v2  (only for phase c/d)')
    parser.add_argument('--config',      type=str,   default='',
                        help='C0..C10 | D0..D6  (only for v1)')
    args = parser.parse_args()

    betas      = args.betas
    start      = args.start_frame
    n_frames   = args.frames
    base_seed  = args.base_seed
    phase      = args.phase.lower()
    variant    = args.variant.lower()
    config     = args.config.upper()

    # Validate
    if phase not in ('raw', 'c', 'd'):
        raise ValueError(f'--phase must be raw/c/d, got {phase}')
    if phase != 'raw' and variant not in ('v1', 'v2'):
        raise ValueError(f'--variant must be v1/v2, got {variant}')
    if phase == 'c' and variant == 'v1' and config not in PHASE_C_V1:
        raise ValueError(f'--config must be one of {list(PHASE_C_V1.keys())}')
    if phase == 'd' and variant == 'v1' and config not in PHASE_D_V1:
        raise ValueError(f'--config must be one of {list(PHASE_D_V1.keys())}')

    artifact_path = resolve_artifact(phase, variant, config)

    print('=' * 72)
    print('Step 6 — Enhancement (variable noise seed = base_seed + frame_i)')
    print('=' * 72)
    print(f'  phase      : {phase}  variant={variant}  config={config or "n/a"}')
    print(f'  betas      : {betas}')
    print(f'  frames     : {start} .. {start + n_frames - 1}')
    print(f'  base_seed  : {base_seed}  (noise seed = base_seed + frame_i)')
    print(f'  results    : {RESULTS}')
    print(f'  log        : {RESULTS}/{_LOG_NAME}')
    print(f'  artifact   : {artifact_path}')
    if not artifact_path.exists():
        raise FileNotFoundError(f'Artifact not found: {artifact_path}  — run step05b first')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    pipeline, flow_model, coords, N_vox = load_pipeline_and_coords()
    flow_model.eval()
    print(f'\n  flow_model: {type(flow_model).__name__}  blocks={flow_model.num_blocks}  '
          f'heads={flow_model.blocks[0].cross_attn.num_heads}  '
          f'qk_rms_norm={flow_model.blocks[0].cross_attn.qk_rms_norm}')

    # ── Load alignment artifact ───────────────────────────────────────────────
    print(f'\n[ALIGN] Loading {artifact_path}...')
    vox_to_tok = torch.load(artifact_path, weights_only=True)
    NVP = vox_to_tok.shape[0]
    print(f'  vox_to_tok shape={tuple(vox_to_tok.shape)}  dtype={vox_to_tok.dtype}')
    print(f'  NVP={NVP}  range=[{int(vox_to_tok.min())}, {int(vox_to_tok.max())}]  '
          f'unique={vox_to_tok.unique().numel()}')

    # ── Build bias_map ────────────────────────────────────────────────────────
    bias_map = torch.zeros(NVP, 1374, dtype=torch.float32)
    bias_map[torch.arange(NVP), vox_to_tok] = 1.0
    print(f'  bias_map shape={tuple(bias_map.shape)}  nonzero={int((bias_map>0).sum())}  '
          f'row_sum mean={bias_map.sum(1).mean():.4f}')

    # ── Encode raw tokens for all needed frames ───────────────────────────────
    # Phase C needs frame i+1; Phase D needs frame i-1 and i+1
    encode_start = max(1, start - (1 if phase == 'd' else 0))
    encode_end   = min(N_FRAMES, start + n_frames - 1 + (1 if phase in ('c', 'd') else 0))

    print(f'\n[ENCODE] Encoding raw DINOv2 tokens for frames {encode_start}..{encode_end}...')
    dino = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(encode_start, encode_end + 1):
        tok = encode_frame_raw(dino, i)
        raw_tokens[i] = tok.cpu()
        if (i - encode_start) % 10 == 0:
            print(f'  frame {i:04d}  mean={tok.float().mean():.5f}  std={tok.float().std():.5f}')
    dino.cpu()
    torch.cuda.empty_cache()
    print(f'  Done. {len(raw_tokens)} frames encoded.')

    # ── Apply smoothing to get conditioned tokens ─────────────────────────────
    print(f'\n[SMOOTH] Applying smoothing: phase={phase}  variant={variant}  config={config or "n/a"}')
    all_tokens = {}   # frame_i → smoothed (1374, 1024) on CPU

    for i in range(start, start + n_frames):
        tok_curr = raw_tokens[i].to(DEVICE)

        if phase == 'raw':
            smoothed = tok_curr

        elif phase == 'c' and variant == 'v1':
            cfg      = PHASE_C_V1[config]
            tok_next = raw_tokens[min(N_FRAMES, i+1)].to(DEVICE)
            smoothed = smooth_c_v1(tok_curr, tok_next, cfg['lc'], cfg['ln'])

        elif phase == 'c' and variant == 'v2':
            tok_next        = raw_tokens[min(N_FRAMES, i+1)].to(DEVICE)
            smoothed, attn_w = smooth_c_v2(tok_curr, tok_next)
            if i == start:
                aw = attn_w.mean(0)
                print(f'  [C v2 wire frame {i}]  attn curr={aw[0]:.4f}  next={aw[1]:.4f}')

        elif phase == 'd' and variant == 'v1':
            cfg      = PHASE_D_V1[config]
            tok_prev = raw_tokens[max(1, i-1)].to(DEVICE)
            tok_next = raw_tokens[min(N_FRAMES, i+1)].to(DEVICE)
            smoothed = smooth_d_v1(tok_prev, tok_curr, tok_next, cfg['lp'], cfg['lc'], cfg['ln'])

        elif phase == 'd' and variant == 'v2':
            tok_prev        = raw_tokens[max(1, i-1)].to(DEVICE)
            tok_next        = raw_tokens[min(N_FRAMES, i+1)].to(DEVICE)
            smoothed, attn_w = smooth_d_v2(tok_prev, tok_curr, tok_next)
            if i == start:
                aw = attn_w.mean(0)
                print(f'  [D v2 wire frame {i}]  attn prev={aw[0]:.4f}  curr={aw[1]:.4f}  next={aw[2]:.4f}')

        # wire check on first frame
        if i == start:
            diff = (smoothed - tok_curr).float()
            print(f'  [SMOOTH wire frame {i}]  smoothed mean={smoothed.float().mean():.5f}  '
                  f'std={smoothed.float().std():.5f}')
            print(f'  [SMOOTH wire frame {i}]  diff vs raw: mean={diff.mean():.5f}  '
                  f'max={diff.abs().max():.5f}  '
                  f'{"(zero = raw, no smoothing)" if diff.abs().max() < 1e-6 else "(non-zero = smoothing applied OK)"}')

        all_tokens[i] = smoothed.cpu()

    pipeline.models['slat_decoder_mesh'].to(DEVICE)

    # ── Run one beta at a time ────────────────────────────────────────────────
    for beta in betas:
        beta_tag = f'beta{beta:.1f}'.replace('.', 'p')
        out_dir  = RESULTS / beta_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*72}')
        print(f'BETA = {beta}  →  {out_dir}')
        print(f'{"="*72}')

        wire_check_log = []
        if beta == 0.0:
            saved = None
            print(f'  beta=0 → original cross-attn  |  phase={phase}  variant={variant}  config={config or "n/a"}')
        else:
            saved = patch_flow_model(flow_model, bias_map, beta, wire_check_log)

        beta_t0      = time.time()
        slat_means   = []
        slat_stds    = []
        render_means = []

        for frame_i in range(start, start + n_frames):
            cond_gl   = all_tokens[frame_i].to(DEVICE)   # (1, 1374, 1024) or (1374, 1024)
            if cond_gl.dim() == 2:
                cond_gl = cond_gl.unsqueeze(0)
            frame_num = frame_i - start + 1

            torch.manual_seed(base_seed + frame_i)
            noise_sp = sp.SparseTensor(
                feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
                coords=coords,
            )

            t0   = time.time()
            x0   = denoise(flow_model, noise_sp, cond_gl)
            slat = normalize_slat(x0)
            del noise_sp, x0

            slat_m = float(slat.feats.float().mean())
            slat_s = float(slat.feats.float().std())
            slat_means.append(slat_m)
            slat_stds.append(slat_s)

            rendered = render(pipeline, slat)
            del slat, cond_gl
            torch.cuda.empty_cache()

            px_mean = float(rendered.mean()) / 255.0
            px_min  = float(rendered.min())  / 255.0
            px_max  = float(rendered.max())  / 255.0
            render_means.append(px_mean)

            out_path = out_dir / f'frame_{frame_i:04d}.png'
            Image.fromarray(rendered).save(out_path)

            if frame_num == 1 and wire_check_log:
                wc = wire_check_log[0]
                print(f'  [WIRE CHECK blk0 first fwd]')
                print(f'    attn@assigned before={wc["before"]:.6f}  after={wc["after"]:.6f}  '
                      f'delta=+{wc["delta"]:.6f}  ratio={wc["after"]/max(wc["before"],1e-9):.3f}x')
                print(f'    attn@rest/tok before={wc["rest_before"]:.8f}  after={wc["rest_after"]:.8f}')

            print(f'  [{frame_num:03d}/{n_frames:03d}] frame={frame_i:04d}'
                  f'  slat_mean={slat_m:.5f}  slat_std={slat_s:.5f}'
                  f'  px_mean={px_mean:.4f}  px_min={px_min:.4f}  px_max={px_max:.4f}'
                  f'  t={time.time()-t0:.1f}s')

        if saved is not None:
            unpatch_flow_model(saved)

        beta_elapsed = time.time() - beta_t0
        print(f'\n  [BETA SUMMARY] beta={beta}  frames={n_frames}  '
              f'total={beta_elapsed:.1f}s  avg={beta_elapsed/max(n_frames,1):.1f}s/frame')
        print(f'  slat_mean: min={min(slat_means):.5f}  max={max(slat_means):.5f}  '
              f'avg={sum(slat_means)/len(slat_means):.5f}')
        print(f'  slat_std : min={min(slat_stds):.5f}   max={max(slat_stds):.5f}   '
              f'avg={sum(slat_stds)/len(slat_stds):.5f}')
        print(f'  px_mean  : min={min(render_means):.4f}  max={max(render_means):.4f}  '
              f'avg={sum(render_means)/len(render_means):.4f}'
              f'  (degenerate if >0.95 or <0.05)')

        if n_frames > 1:
            assemble_video(out_dir, start)

    print(f'\n{"="*72}')
    print('Done.')
    print(f'Results : {RESULTS}')
    print(f'Log     : {RESULTS}/{_LOG_NAME}')
    print(f'{"="*72}')


if __name__ == '__main__':
    main()
