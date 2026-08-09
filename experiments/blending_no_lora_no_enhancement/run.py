"""
blending_no_lora_no_enhancement — MCFM v2 / v3, no enhancement, no LoRA.

Modes (--mode):
  v1_C   naive weighted average, window [i, i+1]
  v1_D   naive weighted average, window [i-1, i, i+1]
  v2_C   per-position temporal attention,  window [i, i+1]
  v2_D   per-position temporal attention,  window [i-1, i, i+1]
  v2b_C  spatial-then-temporal,            window [i, i+1]
  v2b_D  spatial-then-temporal,            window [i-1, i, i+1]
  v3_C   joint temporal+spatial,           window [i, i+1]
  v3_D   joint temporal+spatial,           window [i-1, i, i+1]

BACKWARD COMPATIBLE: existing modes v1/v2/v2b untouched, same math.
New: v3_C, v3_D added. --seed arg (default 6). Tee logging.

Usage:
  # single frame test first:
  python run.py --mode v2_C --start_frame 75 --frames 1 --seed 6
  python run.py --mode v3_C --start_frame 75 --frames 1 --seed 6

  # full 150-frame run:
  python run.py --mode v2_C --frames 150 --seed 6
  sbatch submit_mcfm.sh --mode v2_C
"""

# ── Tee BEFORE all imports ────────────────────────────────────────────────────
import sys, os, argparse as _ap
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

_HERE = Path(__file__).resolve().parent

# Early arg parse for log path only
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--mode',        type=str, default='v2_C')
_pre.add_argument('--seed',        type=int, default=6)
_pre.add_argument('--start_frame', type=int, default=1)
_pre.add_argument('--frames',      type=int, default=None)
_PARGS, _ = _pre.parse_known_args()

_RESULTS_DIR = _HERE / 'results'
_OUT_DIR     = _RESULTS_DIR / f'{_PARGS.mode}_seed{_PARGS.seed}'
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_nf_sfx   = f'n{_PARGS.frames}' if _PARGS.frames is not None else 'nall'
_LOG_PATH = _OUT_DIR / f'run_f{_PARGS.start_frame:04d}_{_nf_sfx}.log'


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg)
        self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()

sys.stdout = _Tee(_LOG_PATH)
sys.stderr = sys.stdout

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse, math, time
import numpy as np
import torch
from PIL import Image
import subprocess

_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp

from step1_input_prep.input_prep       import load_frame, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step4_mcfm.mcfm                   import mcfm_v1, mcfm_v2, mcfm_v2b, mcfm_v3

# ── Constants ──────────────────────────────────────────────────────────────────
PRETRAINED  = 'microsoft/TRELLIS-image-large'
GT_FRAME_75 = str(_PIPE / '..' / '..' / '..' /
                  'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                  '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
DEVICE      = torch.device('cuda')
STEPS       = 25
RENDER_RES  = 518
RESCALE_T   = 3.0
STRUCT_SEED = 42   # voxel structure seed — never changes
_SCALE_DIAG = 1024 ** -0.5

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

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]

_fx_n      = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_fx_n, 0., 0.5], [0., _fx_n, 0.5], [0., 0., 1.]],
                           dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.tensor([
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  1.,  0.,  2.],
    [ 0.,  0.,  0.,  1.],
], dtype=torch.float32, device=DEVICE)

# ── Mode definitions ──────────────────────────────────────────────────────────
MODES = {
    'v1_C':  dict(fn=mcfm_v1,  window='C'),
    'v1_D':  dict(fn=mcfm_v1,  window='D'),
    'v2_C':  dict(fn=mcfm_v2,  window='C'),
    'v2_D':  dict(fn=mcfm_v2,  window='D'),
    'v2b_C': dict(fn=mcfm_v2b, window='C'),
    'v2b_D': dict(fn=mcfm_v2b, window='D'),
    'v3_C':  dict(fn=mcfm_v3,  window='C'),   # NEW
    'v3_D':  dict(fn=mcfm_v3,  window='D'),   # NEW
}


def build_window(frame_i, window_type):
    if window_type == 'C':
        indices = [frame_i, min(N_FRAMES, frame_i + 1)]
        weights = torch.tensor([0.5, 0.5])
    else:  # D
        indices = [max(1, frame_i - 1), frame_i, min(N_FRAMES, frame_i + 1)]
        weights = torch.tensor([1/3, 1/3, 1/3])   # uniform per spec (was 0.25/0.50/0.25)
    return indices, weights


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def _ts(t, label):
    """Log tensor stats."""
    f = t.float()
    print(f'  [{label}] shape={tuple(t.shape)} dtype={t.dtype} '
          f'mean={f.mean():.5f} std={f.std():.5f} '
          f'min={f.min():.5f} max={f.max():.5f}')


def _diag_v2_attn(gpu_tokens, win_indices, frame_idx_t):
    """Recompute v2 attention weights for diagnostic logging only."""
    all_tok = torch.stack([gpu_tokens[i]['tokens'] for i in win_indices])  # (W, 1374, 1024)
    q_t     = gpu_tokens[frame_idx_t]['tokens']                             # (1374, 1024)
    Q = q_t.unsqueeze(1)                                                    # (1374, 1, 1024)
    K = all_tok.permute(1, 0, 2)                                            # (1374, W, 1024)
    scores = torch.bmm(Q, K.transpose(1, 2)) * _SCALE_DIAG                 # (1374, 1, W)
    attn   = torch.softmax(scores, dim=-1).squeeze(1)                       # (1374, W)
    W = attn.shape[1]
    # find index of frame_idx_t in win_indices
    try:
        curr_idx = win_indices.index(frame_idx_t)
    except ValueError:
        curr_idx = 0
    attn_mean = attn.mean(0)  # (W,)
    pct_curr  = (attn.argmax(1) == curr_idx).float().mean().item() * 100
    print(f'  [ATTN-V2] per-frame mean attn: {[f"{attn_mean[i].item():.4f}" for i in range(W)]}')
    print(f'  [ATTN-V2] sum={attn_mean.sum():.4f}  curr_idx={curr_idx}  '
          f'curr_wins_at={pct_curr:.1f}% of 1374 positions '
          f'{"✓" if pct_curr > (100/W) else "✗ curr not dominant"}')
    print(f'  [ATTN-V2] curr attn per-pos: min={attn[:,curr_idx].min():.4f}  '
          f'max={attn[:,curr_idx].max():.4f}  std={attn[:,curr_idx].std():.4f}')


def _diag_v3_attn(gpu_tokens, win_indices, frame_idx_t):
    """Recompute v3 attention weights for diagnostic logging only."""
    all_tok = torch.stack([gpu_tokens[i]['tokens'] for i in win_indices])  # (W, 1374, 1024)
    pool    = all_tok.view(-1, 1024)                                        # (W*1374, 1024)
    q_t     = gpu_tokens[frame_idx_t]['tokens']                             # (1374, 1024)
    scores  = torch.mm(q_t, pool.T) * _SCALE_DIAG                          # (1374, W*1374)
    attn    = torch.softmax(scores, dim=-1)                                 # (1374, W*1374)
    W       = len(win_indices)
    try:
        curr_idx = win_indices.index(frame_idx_t)
    except ValueError:
        curr_idx = 0
    # mass going to each frame's token block
    per_frame_mass = attn.view(1374, W, 1374).sum(dim=2).mean(0)  # (W,) mean over query positions
    curr_mass = per_frame_mass[curr_idx].item()
    print(f'  [ATTN-V3] mean attn mass per frame: {[f"{per_frame_mass[i].item():.4f}" for i in range(W)]}')
    print(f'  [ATTN-V3] sum={per_frame_mass.sum():.4f}  curr_frame_mass={curr_mass:.4f}  '
          f'(uniform would be {1/W:.4f})  '
          f'{"✓ curr dominant" if curr_mass > 1/W else "✗ curr below uniform"}')
    # cross-position mass: how much does position j attend outside its own position?
    same_pos_mass = attn.view(1374, W, 1374)[:, :, :].diagonal(dim1=0, dim2=2)  # not straightforward
    # simpler: for each query pos, how much goes to the SAME position across all frames
    # attn[j, frame*1374 + j] for all frames
    same_pos = sum(attn[j, f * 1374 + j].item() for j in range(1374) for f in range(W)) / 1374
    print(f'  [ATTN-V3] mean mass on same-position tokens: {same_pos:.4f}  '
          f'cross-position mass: {1.0 - same_pos:.4f}')


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def normalize_slat(x0):
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    return x0.replace(x0.feats * std + mean)


def denoise(flow_model, noise_sp, cond_gl):
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = flow_model(x, t_ten, cond_gl)
            x = x.replace(x.feats - (t - t_prev) * v.feats)
    return x


def render_slat(pipeline, slat, renderer):
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    mask  = result['mask'].unsqueeze(0)
    color = result['color'] * mask + (1.0 - mask)
    coverage = mask.float().mean().item() * 100
    return (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), coverage


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        type=str, required=True, choices=list(MODES))
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--frames',      type=int, default=None)
    parser.add_argument('--seed',        type=int, default=6,
                        help='Fixed noise seed — same noise every frame (default 6)')
    args = parser.parse_args()

    cfg      = MODES[args.mode]
    blend_fn = cfg['fn']
    win_type = cfg['window']
    seed     = args.seed
    start    = args.start_frame
    n_frames = args.frames or N_FRAMES

    print('=' * 72)
    print(f'blending_no_lora_no_enhancement  —  MCFM {args.mode}')
    print('=' * 72)
    print(f'  mode        : {args.mode}')
    print(f'  window      : {win_type}  ({"[i, i+1]" if win_type == "C" else "[i-1, i, i+1]"})')
    print(f'  seed        : {seed}  (fixed — same noise every frame)')
    print(f'  frames      : {start} .. {start + n_frames - 1}')
    print(f'  output dir  : {_OUT_DIR}')
    print(f'  log         : {_LOG_PATH}')
    print(f'  STEPS       : {STEPS}')
    print(f'  STRUCT_SEED : {STRUCT_SEED}')
    print(f'  DEVICE      : {DEVICE}')

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\n[LOAD] Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    print(f'  flow_model.in_channels = {flow_model.in_channels}')

    print(f'\n[STRUCT] Sampling voxel structure (frame 75, STRUCT_SEED={STRUCT_SEED})...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  N_vox  = {N_vox}')
    print(f'  coords shape = {tuple(coords.shape)}')
    print(f'  coords sum   = {coords.sum().item()}  (hash proxy for reproducibility — same seed must give same value)')

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── Pre-encode all frames ─────────────────────────────────────────────────
    print(f'\n[DINO] Pre-encoding {N_FRAMES} frames with DINOv2...')
    dino = load_dino(DEVICE)
    t0   = time.time()
    all_tokens = {}
    for i in range(1, N_FRAMES + 1):
        toks = encode_frames([i], [load_frame(i)], dino, DEVICE)
        all_tokens[i] = {k: v.cpu() for k, v in toks[i].items()}
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    print(f'  Done in {time.time()-t0:.1f}s')
    # spot-check frame 75 tokens
    t75 = all_tokens[75]['tokens']
    print(f'  [SPOT-CHECK frame 75] shape={tuple(t75.shape)}  '
          f'mean={t75.float().mean():.5f}  std={t75.float().std():.5f}  '
          f'(post-layernorm: expect mean~0)')
    del dino
    torch.cuda.empty_cache()

    renderer   = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )
    flow_model.eval()

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'\n[NOISE] Building fixed noise (seed={seed})...')
    torch.manual_seed(seed)
    fixed_noise_feats = torch.randn(N_vox, flow_model.in_channels, device=DEVICE)
    print(f'  shape={tuple(fixed_noise_feats.shape)}  '
          f'mean={fixed_noise_feats.mean():.5f}  std={fixed_noise_feats.std():.5f}')
    print(f'  Same tensor cloned every frame — isolates noise as flicker source.')

    # ── Per-frame loop ────────────────────────────────────────────────────────
    print(f'\n[TRAIN] Rendering {n_frames} frame(s)...\n')
    t0_total = time.time()

    for frame_i in range(start, start + n_frames):
        t_frame = time.time()
        win_indices, lambda_vec = build_window(frame_i, win_type)

        print(f'{"="*60}')
        print(f'[FRAME {frame_i:04d}]  window={win_indices}  lambda={lambda_vec.tolist()}')

        # ── STEP 2 wire-check: token stats ───────────────────────────────────
        gpu_tokens = {idx: {'tokens': all_tokens[idx]['tokens'].to(DEVICE)}
                      for idx in set(win_indices)}
        for idx in win_indices:
            _ts(gpu_tokens[idx]['tokens'], f'STEP2 tok[{idx:04d}]')

        # duplicate-frame check at boundaries
        unique = set(win_indices)
        if len(unique) < len(win_indices):
            dups = [i for i in win_indices if win_indices.count(i) > 1]
            print(f'  [STEP1] boundary repeat: {dups} — expected, increases weight on boundary frame')

        # ── STEP 4 wire-check: blend ──────────────────────────────────────────
        K_hat, _ = blend_fn(gpu_tokens, win_indices, frame_i, lambda_vec.to(DEVICE))
        cond_gl  = K_hat.unsqueeze(0)   # (1, 1374, 1024)

        _ts(K_hat,   'STEP4 K_hat  ')
        print(f'  [STEP4] cond_gl shape={tuple(cond_gl.shape)}  dtype={cond_gl.dtype}  → flow model input')

        # diff K_hat vs raw current-frame token
        diff = (K_hat - gpu_tokens[frame_i]['tokens']).float()
        print(f'  [STEP4] K_hat vs tok[curr]:  diff_mean={diff.mean():.6f}  '
              f'diff_max={diff.abs().max():.6f}  '
              f'{"identical — no blend effect" if diff.abs().max() < 1e-5 else "blending active ✓"}')

        # attention diagnostics (v2 and v3 only)
        if 'v2' in args.mode and 'v2b' not in args.mode:
            _diag_v2_attn(gpu_tokens, win_indices, frame_i)
        elif 'v3' in args.mode:
            _diag_v3_attn(gpu_tokens, win_indices, frame_i)

        # ── STEP 5 wire-check: noise ──────────────────────────────────────────
        noise_sp = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
        _ts(noise_sp.feats, 'STEP5 noise  ')

        # ── Denoise ───────────────────────────────────────────────────────────
        x0 = denoise(flow_model, noise_sp, cond_gl)
        _ts(x0.feats, 'STEP5 x0     ')
        del noise_sp

        # ── SLaT normalise ────────────────────────────────────────────────────
        slat = normalize_slat(x0)
        del x0
        _ts(slat.feats, 'STEP6 slat   ')

        # ── Render ────────────────────────────────────────────────────────────
        rendered, coverage = render_slat(pipeline, slat, renderer)
        del slat
        torch.cuda.empty_cache()

        print(f'  [STEP7] rendered shape={rendered.shape}  dtype={rendered.dtype}  '
              f'min={rendered.min()}  max={rendered.max()}  '
              f'mask_coverage={coverage:.1f}%  '
              f'{"✓" if 5 < coverage < 95 else "✗ degenerate render"}')

        out_path = _OUT_DIR / f'frame_{frame_i:04d}.png'
        Image.fromarray(rendered).save(out_path)

        elapsed = time.time() - t_frame
        print(f'  [DONE]  frame {frame_i:04d}  elapsed={elapsed:.1f}s  saved={out_path.name}')

        del K_hat, cond_gl, gpu_tokens

    # ── Video ─────────────────────────────────────────────────────────────────
    total_min = (time.time() - t0_total) / 60
    print(f'\n[SUMMARY] {n_frames} frames in {total_min:.1f} min  '
          f'({total_min*60/n_frames:.1f}s/frame)')

    if n_frames > 1:
        vid_path = _OUT_DIR / 'render.mp4'
        print(f'[VIDEO] Assembling -> {vid_path}')
        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-framerate', '25',
            '-i', str(_OUT_DIR / 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid_path)
        ], check=True)
        print(f'[VIDEO] Done: {vid_path}')
    else:
        print('[VIDEO] Single frame — skipping video assembly.')

    print(f'\n[DONE] Log: {_LOG_PATH}')
    print(f'[DONE] Results: {_OUT_DIR}')


if __name__ == '__main__':
    main()
