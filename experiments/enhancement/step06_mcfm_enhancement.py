"""
Step 6 MCFM Enhancement — bias injection on top of MCFM v2/v3 blended tokens.

Modes (--mode):
  v2_C  — per-position temporal attn, 2-frame window [i, i+1]
  v2_D  — per-position temporal attn, 3-frame window [i-1, i, i+1]
  v3_C  — joint temporal+spatial attn, 2-frame window [i, i+1]
  v3_D  — joint temporal+spatial attn, 3-frame window [i-1, i, i+1]

BACKWARD COMPATIBLE:
  step06_enhancement.py is NOT modified — raw / phase_c / phase_d modes still use it.
  Same artifact (artifacts/voxel_to_token.pt, shape=(1748,)), same bias injection logic,
  same STRUCT_SEED=42, same SLAT_MEAN/STD/EXTRINSICS/INTRINSICS.

  Artifact note: voxel_to_token maps the 1748 voxels that reach the cross-attn
  layer (after sparse-conv downsampling from 7301 input voxels). Both this script
  and the blending runs start from N_vox=7301, so the artifact is compatible.

Fixed noise by default (seed=6 every frame) — matches blending_no_lora_no_enhancement.
Pass --variable_noise to use base_seed + frame_i per frame.

Usage:
  # single-frame sanity check
  python step06_mcfm_enhancement.py --mode v2_C --betas 0 2 4 8 --start_frame 75 --frames 1

  # full 150-frame run
  python step06_mcfm_enhancement.py --mode v2_C --betas 0 2 4 8 --start_frame 1 --frames 150
  python step06_mcfm_enhancement.py --mode v2_D --betas 0 2 4 8 --start_frame 1 --frames 150
  python step06_mcfm_enhancement.py --mode v3_C --betas 0 2 4 8 --start_frame 1 --frames 150
  python step06_mcfm_enhancement.py --mode v3_D --betas 0 2 4 8 --start_frame 1 --frames 150
"""

import sys, os, time, argparse as _ap
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'

# ── Early arg parse — must happen before Tee so log path is known ─────────────
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode',          type=str,   default='v2_C',
                  choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
_pre.add_argument('--betas',         type=float, nargs='+', default=[0.0, 2.0])
_pre.add_argument('--base_seed',     type=int,   default=6)
_pre.add_argument('--start_frame',   type=int,   default=75)
_pre.add_argument('--frames',        type=int,   default=1)
_pre.add_argument('--variable_noise', action='store_true')
_PRE, _ = _pre.parse_known_args()

_noise_sfx  = 'varnoise' if _PRE.variable_noise else 'fixednoise'
_exp_tag    = f'mcfm_{_PRE.mode}_seed{_PRE.base_seed}_{_noise_sfx}'
RESULTS     = _HERE / f'results_{_exp_tag}'
RESULTS.mkdir(parents=True, exist_ok=True)
_betas_str  = '_'.join(str(int(b) if b == int(b) else b) for b in _PRE.betas)
_LOG_NAME   = f'run_f{_PRE.start_frame:04d}_n{_PRE.frames}_betas{_betas_str}.log'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()


sys.stdout = _Tee(RESULTS / _LOG_NAME)
sys.stderr = sys.stdout

# ── Imports after Tee (their prints go to log too) ────────────────────────────
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
from step4_mcfm.mcfm import mcfm_v2, mcfm_v3

# ── Constants — identical to step06_enhancement.py ───────────────────────────
GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150').resolve()
GT_FRAME_75   = GT_FRAMES_DIR / 'frame_0075.png'

PRETRAINED   = 'microsoft/TRELLIS-image-large'
DEVICE       = torch.device('cuda')
STRUCT_SEED  = 42
N_FRAMES     = 150
RENDER_RES   = 518
RESCALE_T    = 3.0
STEPS        = 25
ARTIFACTS    = _HERE / 'artifacts'

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i+1]) for i in range(STEPS)]

_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
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
    -1.1414450407028198,  1.2039363384246826,
], dtype=torch.float32)
SLAT_STD = torch.tensor([
    2.377650737762451, 2.386378288269043, 2.124418020248413,
    2.1748552322387695, 2.663944721221924, 2.371192216873169,
    2.6217446327209473, 2.684523105621338,
], dtype=torch.float32)

_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_SCALE     = 1024 ** -0.5


# ── Window helpers ─────────────────────────────────────────────────────────────

def get_window_indices(frame_idx: int, mode: str) -> list:
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    else:
        return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def get_lambda_vec(window_indices: list) -> torch.Tensor:
    n = len(window_indices)
    return torch.tensor([1.0 / n] * n, dtype=torch.float32, device=DEVICE)


# ── MCFM blend ─────────────────────────────────────────────────────────────────

def apply_mcfm(mode: str, raw_tokens: dict, frame_idx: int):
    """Blend tokens for frame_idx. Returns (K_hat, win, lambda_vec)."""
    win = get_window_indices(frame_idx, mode)
    lam = get_lambda_vec(win)
    # mcfm.py format: {frame_idx: {'tokens': (1374, 1024)}}
    tok_dict = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn  = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_dict, win, frame_idx, lam)
    return K_hat, win, lam


# ── MCFM diagnostic logging ────────────────────────────────────────────────────

def log_mcfm(mode: str, raw_tokens: dict, frame_idx: int, K_hat: torch.Tensor, win: list):
    tok_curr  = raw_tokens[frame_idx].to(DEVICE)
    diff      = (K_hat.float() - tok_curr.float()).abs()
    is_active = diff.max().item() > 1e-6
    all_tok   = torch.stack([raw_tokens[i].to(DEVICE) for i in win], dim=0)  # (W, 1374, 1024)
    W         = len(win)
    curr_idx  = win.index(frame_idx)

    # log boundary repeats
    seen = [win[0]]
    for w in win[1:]:
        if w == seen[-1]:
            print(f'  [STEP1] boundary repeat: [{seen[-1]}, {w}] — '
                  f'expected, doubles weight on boundary frame')
        seen.append(w)

    print(f'  [MCFM] window={win}  lambda=[{1/W:.4f}]*{W}  mode={mode}')
    print(f'  [MCFM] K_hat: shape={tuple(K_hat.shape)}  mean={K_hat.float().mean():.5f}  '
          f'std={K_hat.float().std():.5f}')
    print(f'  [MCFM] K_hat vs tok[curr]: diff_mean={diff.mean():.6f}  '
          f'diff_max={diff.max():.6f}  '
          f'{"blending active ✓" if is_active else "identical — no blend (boundary degenerate)"}')

    with torch.no_grad():
        if mode.startswith('v2'):
            Q      = tok_curr.unsqueeze(1)                              # (1374, 1, 1024)
            K      = all_tok.permute(1, 0, 2)                          # (1374, W, 1024)
            attn   = torch.softmax(
                torch.bmm(Q, K.transpose(1, 2)) * _SCALE, dim=-1
            ).squeeze(1)                                                # (1374, W)
            m_attn = attn.mean(0)
            pct    = (attn.argmax(1) == curr_idx).float().mean().item() * 100
            print(f'  [ATTN-V2] per-frame mean: {[f"{v:.4f}" for v in m_attn.tolist()]}  '
                  f'sum={m_attn.sum():.4f}')
            print(f'  [ATTN-V2] curr_idx={curr_idx}  curr_wins_at={pct:.1f}%  '
                  f'curr_per_pos min={attn[:,curr_idx].min():.4f}  '
                  f'max={attn[:,curr_idx].max():.4f}  std={attn[:,curr_idx].std():.4f}')
        else:  # v3
            N    = 1374
            pool = all_tok.view(-1, 1024)                               # (W*N, 1024)
            attn = torch.softmax(
                torch.mm(tok_curr, pool.T) * _SCALE, dim=-1
            )                                                           # (N, W*N)
            mass = attn.view(N, W, N).sum(2).mean(0)                   # (W,) per-frame
            # same-position: attn[j, k*N + j] for each frame k
            same_pos = torch.zeros(N, device=DEVICE)
            for k in range(W):
                same_pos += attn[torch.arange(N, device=DEVICE),
                                 k * N + torch.arange(N, device=DEVICE)]
            same_total  = same_pos.mean().item()
            cross_total = 1.0 - same_total
            uniform     = 1.0 / W
            print(f'  [ATTN-V3] per-frame mass: {[f"{v:.4f}" for v in mass.tolist()]}  '
                  f'sum={mass.sum():.4f}')
            print(f'  [ATTN-V3] curr_frame_mass={mass[curr_idx]:.4f}  '
                  f'(uniform={uniform:.4f})  '
                  f'{"✓ curr dominant" if mass[curr_idx] >= uniform else "✗ curr below uniform"}')
            print(f'  [ATTN-V3] same-pos mass={same_total:.4f}  '
                  f'cross-pos mass={cross_total:.4f}')


# ── DINOv2 encoder — identical to step06_enhancement.py ───────────────────────

def encode_frame(dino_model, frame_idx: int) -> torch.Tensor:
    img_path = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    img  = Image.open(img_path).convert('RGB').resize((518, 518), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32) / 255.0
    x    = _DINO_NORM(
        torch.from_numpy(arr).permute(2, 0, 1).float()
    ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = dino_model(x, is_training=True)['x_prenorm']
        tokens = F.layer_norm(feats, feats.shape[-1:])
    return tokens.squeeze(0)   # (1374, 1024)


# ── Cross-attn patch — identical math to step06_enhancement.py ────────────────

def patch_flow_model(flow_model, bias_map, beta, wire_log=None):
    """Inject additive bias into all cross-attn blocks. Returns list of (module, orig_fwd)."""
    bm    = bias_map.to(DEVICE)
    saved = []
    for blk_idx, block in enumerate(flow_model.blocks):
        ca   = block.cross_attn
        nh   = ca.num_heads
        hd   = ca.channels // nh
        sc   = hd ** -0.5
        orig = ca.forward

        def make_fwd(mod, nh_, hd_, sc_, bias, bt, blk_i, wl):
            fired = [False]
            def biased_fwd(x_st, context=None):
                nv      = x_st.feats.shape[0]
                w_dt    = mod.to_q.weight.dtype
                q       = mod.to_q(x_st.feats.to(w_dt)).reshape(nv, nh_, hd_)
                kv_proj = mod.to_kv(context[0].to(w_dt))
                kv      = kv_proj.reshape(-1, 2, nh_, hd_)
                k, v    = kv[:, 0], kv[:, 1]
                scores_raw = torch.einsum('nhd,mhd->nhm', q.float(), k.float()) * sc_
                scores     = scores_raw + bt * bias.unsqueeze(1).to(scores_raw.dtype)
                if blk_i == 0 and wl is not None and not fired[0]:
                    fired[0] = True
                    with torch.no_grad():
                        ab   = torch.softmax(scores_raw, dim=-1).mean(1)
                        aa   = torch.softmax(scores,     dim=-1).mean(1)
                        asgn = bias.argmax(dim=1)
                        wl.append({
                            'nv':          nv,
                            'before':      ab[torch.arange(nv), asgn].mean().item(),
                            'after':       aa[torch.arange(nv), asgn].mean().item(),
                            'delta':       (aa - ab)[torch.arange(nv), asgn].mean().item(),
                            'rest_before': ((ab.sum(1) - ab[torch.arange(nv), asgn]) / 1373).mean().item(),
                            'rest_after':  ((aa.sum(1) - aa[torch.arange(nv), asgn]) / 1373).mean().item(),
                        })
                attn = torch.softmax(scores, dim=-1).to(w_dt)
                out  = torch.einsum('nhm,mhd->nhd', attn, v).reshape(nv, nh_ * hd_)
                return x_st.replace(mod.to_out(out))
            return biased_fwd

        ca.forward = make_fwd(ca, nh, hd, sc, bm, beta, blk_idx, wire_log)
        saved.append((ca, orig))

    print(f'  [PATCH] beta={beta}  patched {len(saved)} cross_attn blocks')
    print(f'  [PATCH] bias_map device={bm.device}  shape={tuple(bm.shape)}  '
          f'nonzero={int((bm > 0).sum())}  row_sum_mean={bm.sum(1).mean():.4f}')
    return saved


def unpatch_flow_model(saved):
    for ca, orig in saved:
        ca.forward = orig
    print(f'  [UNPATCH] restored {len(saved)} cross_attn blocks')


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def load_pipeline_and_coords():
    print('[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    print(f'\n[STRUCT] Sampling voxel structure (frame 75, STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox  = {N_vox}  (expect 7301 to match blending runs)')
    print(f'  coords shape = {tuple(coords.shape)}')
    print(f'  coords sum   = {coords.sum().item()}  '
          f'(hash proxy for reproducibility — expect 661358)')
    assert N_vox == 7301, f'N_vox={N_vox} != 7301 — voxel structure mismatch with blending runs'

    print(f'\n  flow_model: blocks={flow_model.num_blocks}  '
          f'heads={flow_model.blocks[0].cross_attn.num_heads}  '
          f'in_channels={flow_model.in_channels}')

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
            v     = flow_model(x, t_ten, cond_gl)
            x     = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def normalize_slat(x0):
    return x0.replace(
        x0.feats * SLAT_STD.to(x0.feats.device) + SLAT_MEAN.to(x0.feats.device)
    )


def render_frame(pipeline, slat):
    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )
    with torch.no_grad():
        mesh = pipeline.decode_slat(slat, ['mesh'])['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS, return_types=['color', 'mask'])
    mask   = result['mask'].unsqueeze(0)
    color  = result['color'] * mask + (1.0 - mask)
    return (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def assemble_video(out_dir: Path, start_frame: int):
    import subprocess
    vid = out_dir / 'render.mp4'
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-framerate', '25',
        '-start_number', str(start_frame),
        '-i', str(out_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid),
    ], check=True, capture_output=True)
    print(f'  [VIDEO] Done: {vid}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser()
    parser.add_argument('--mode',          type=str,   default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--betas',         type=float, nargs='+', default=[0.0, 2.0])
    parser.add_argument('--base_seed',     type=int,   default=6)
    parser.add_argument('--start_frame',   type=int,   default=75)
    parser.add_argument('--frames',        type=int,   default=1)
    parser.add_argument('--variable_noise', action='store_true',
                        help='Use base_seed+frame_i per frame instead of fixed seed')
    args = parser.parse_args()

    mode         = args.mode
    betas        = args.betas
    start        = args.start_frame
    n_frames     = args.frames
    base_seed    = args.base_seed
    fixed_noise  = not args.variable_noise

    print('=' * 72)
    print('Step 6 MCFM Enhancement')
    print('=' * 72)
    print(f'  mode         : {mode}')
    print(f'  betas        : {betas}')
    print(f'  frames       : {start} .. {start + n_frames - 1}')
    print(f'  base_seed    : {base_seed}')
    print(f'  noise        : {"fixed (same seed every frame)" if fixed_noise else "variable (base_seed + frame_i)"}')
    print(f'  results      : {RESULTS}')
    print(f'  log          : {RESULTS / _LOG_NAME}')

    artifact_path = ARTIFACTS / 'voxel_to_token.pt'
    print(f'  artifact     : {artifact_path}')
    if not artifact_path.exists():
        raise FileNotFoundError(f'Artifact not found: {artifact_path}  — run step05b first')

    # ── Load pipeline + coords ────────────────────────────────────────────────
    pipeline, flow_model, coords, N_vox = load_pipeline_and_coords()
    flow_model.eval()

    # ── Load artifact + build bias_map ────────────────────────────────────────
    print(f'\n[ALIGN] Loading {artifact_path}...')
    vox_to_tok = torch.load(artifact_path, weights_only=True)
    NVP        = vox_to_tok.shape[0]
    print(f'  vox_to_tok shape={tuple(vox_to_tok.shape)}  dtype={vox_to_tok.dtype}')
    print(f'  NVP={NVP}  range=[{int(vox_to_tok.min())}, {int(vox_to_tok.max())}]  '
          f'unique_tokens={vox_to_tok.unique().numel()}')
    print(f'  Note: NVP={NVP} is the cross-attn voxel count after sparse-conv '
          f'downsampling (from N_vox=7301 input)')

    bias_map = torch.zeros(NVP, 1374, dtype=torch.float32)
    bias_map[torch.arange(NVP), vox_to_tok] = 1.0
    print(f'  bias_map shape={tuple(bias_map.shape)}  '
          f'nonzero={int((bias_map > 0).sum())}  '
          f'row_sum mean={bias_map.sum(1).mean():.4f}')

    # ── Encode all DINOv2 tokens ───────────────────────────────────────────────
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    t_enc = time.time()
    for i in range(1, N_FRAMES + 1):
        tok = encode_frame(dino_model, i)
        raw_tokens[i] = tok.cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}  mean={tok.float().mean():.5f}  std={tok.float().std():.5f}')
    dino_model.cpu()
    torch.cuda.empty_cache()
    print(f'  Done in {time.time() - t_enc:.1f}s')
    # spot-check frame 75
    t75 = raw_tokens[75].float()
    print(f'  [SPOT-CHECK frame 75] shape={tuple(t75.shape)}  '
          f'mean={t75.mean():.5f}  std={t75.std():.5f}  '
          f'(post-layernorm: expect mean~0, std~1)')

    # ── Fixed noise tensor (if fixed_noise mode) ──────────────────────────────
    fixed_noise_feats = None
    if fixed_noise:
        print(f'\n[NOISE] Building fixed noise (seed={base_seed})...')
        torch.manual_seed(base_seed)
        fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
        print(f'  shape={tuple(fixed_noise_feats.shape)}  '
              f'mean={fixed_noise_feats.mean():.5f}  std={fixed_noise_feats.std():.5f}')
        print(f'  Same tensor cloned every frame — isolates noise as flicker source.')

    pipeline.models['slat_decoder_mesh'].to(DEVICE)

    # ── Run one beta at a time ─────────────────────────────────────────────────
    for beta in betas:
        beta_tag = f'beta{beta:.1f}'.replace('.', 'p')
        out_dir  = RESULTS / beta_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"=" * 72}')
        print(f'BETA = {beta}  →  {out_dir}')
        print(f'{"=" * 72}')

        wire_log = []
        if beta == 0.0:
            saved = None
            print(f'  beta=0 → no cross-attn patching  (baseline, mcfm tokens only)')
        else:
            saved = patch_flow_model(flow_model, bias_map, beta, wire_log)

        beta_t0      = time.time()
        slat_means   = []
        slat_stds    = []
        render_means = []

        for frame_i in range(start, start + n_frames):
            frame_num = frame_i - start + 1
            t0 = time.time()

            # ── MCFM blend ────────────────────────────────────────────────────
            K_hat, win, lam = apply_mcfm(mode, raw_tokens, frame_i)
            cond_gl         = K_hat.unsqueeze(0)   # (1, 1374, 1024)

            print(f'\n{"─" * 60}')
            print(f'[FRAME {frame_i:04d}]  ({frame_num}/{n_frames})  beta={beta}')
            log_mcfm(mode, raw_tokens, frame_i, K_hat, win)
            print(f'  [STEP4] cond_gl shape={tuple(cond_gl.shape)}  '
                  f'dtype={cond_gl.dtype}  → flow model input ✓')

            # ── Noise ─────────────────────────────────────────────────────────
            if fixed_noise:
                noise_feats = fixed_noise_feats.clone()
            else:
                torch.manual_seed(base_seed + frame_i)
                noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
            noise_sp = sp.SparseTensor(feats=noise_feats, coords=coords)
            print(f'  [STEP5 noise] shape={tuple(noise_feats.shape)}  '
                  f'mean={noise_feats.mean():.5f}  std={noise_feats.std():.5f}')

            # ── Denoise ───────────────────────────────────────────────────────
            x0   = denoise(flow_model, noise_sp, cond_gl)
            print(f'  [STEP5 x0   ] shape={tuple(x0.feats.shape)}  '
                  f'mean={x0.feats.float().mean():.5f}  std={x0.feats.float().std():.5f}')

            # ── Normalize SLaT ────────────────────────────────────────────────
            slat = normalize_slat(x0)
            del noise_sp, x0
            slat_m = float(slat.feats.float().mean())
            slat_s = float(slat.feats.float().std())
            slat_means.append(slat_m)
            slat_stds.append(slat_s)
            print(f'  [STEP6 slat ] shape={tuple(slat.feats.shape)}  '
                  f'mean={slat_m:.5f}  std={slat_s:.5f}')

            # ── Wire check (first frame of first beta with patching) ──────────
            if frame_num == 1 and wire_log:
                wc = wire_log[0]
                print(f'  [WIRE-CHECK blk0]  nv={wc["nv"]}  '
                      f'attn@assigned before={wc["before"]:.6f}  after={wc["after"]:.6f}  '
                      f'delta=+{wc["delta"]:.6f}  ratio={wc["after"]/max(wc["before"],1e-9):.3f}x')
                print(f'  [WIRE-CHECK blk0]  attn@rest before={wc["rest_before"]:.8f}  '
                      f'after={wc["rest_after"]:.8f}')

            # ── Render ────────────────────────────────────────────────────────
            rendered = render_frame(pipeline, slat)
            del slat, cond_gl, K_hat
            torch.cuda.empty_cache()

            px_mean = float(rendered.mean()) / 255.0
            px_min  = int(rendered.min())
            px_max  = int(rendered.max())
            render_means.append(px_mean)
            print(f'  [STEP7 render] shape={rendered.shape}  dtype={rendered.dtype}  '
                  f'min={px_min}  max={px_max}  px_mean={px_mean:.4f}  '
                  f'{"⚠ DEGENERATE" if px_mean > 0.95 or px_mean < 0.05 else "✓"}')
            print(f'  [DONE] frame {frame_i:04d}  elapsed={time.time()-t0:.1f}s')

            out_path = out_dir / f'frame_{frame_i:04d}.png'
            Image.fromarray(rendered).save(out_path)
            print(f'  saved: {out_path.name}')

        if saved is not None:
            unpatch_flow_model(saved)

        beta_elapsed = time.time() - beta_t0
        print(f'\n[BETA SUMMARY] beta={beta}  frames={n_frames}  '
              f'total={beta_elapsed:.1f}s  avg={beta_elapsed/max(n_frames,1):.1f}s/frame')
        print(f'  slat_mean : min={min(slat_means):.5f}  max={max(slat_means):.5f}  '
              f'avg={sum(slat_means)/len(slat_means):.5f}')
        print(f'  slat_std  : min={min(slat_stds):.5f}  max={max(slat_stds):.5f}  '
              f'avg={sum(slat_stds)/len(slat_stds):.5f}')
        print(f'  px_mean   : min={min(render_means):.4f}  max={max(render_means):.4f}  '
              f'avg={sum(render_means)/len(render_means):.4f}  '
              f'(⚠ degenerate if >0.95 or <0.05)')

        if n_frames > 1:
            print(f'\n[VIDEO] Assembling...')
            assemble_video(out_dir, start)

    print(f'\n{"=" * 72}')
    print('Done.')
    print(f'Results : {RESULTS}')
    print(f'Log     : {RESULTS / _LOG_NAME}')
    print(f'{"=" * 72}')


if __name__ == '__main__':
    main()
