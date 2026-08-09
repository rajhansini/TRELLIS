"""
Render all 150 frames with the best LoRA checkpoint from step07d.

For each run, saves:
  results_dir/full_render/ctrl/frame_XXXX.png    (alpha=0 MCFM baseline)
  results_dir/full_render/lora/frame_XXXX.png    (LoRA active)
  results_dir/full_render/comp/frame_XXXX.png    (GT | ctrl | lora side-by-side)
  results_dir/full_render/comparison.mp4         (video)

Resumable: skips frames already saved.

Usage:
  python step07d_render_best.py --alpha 0.1  --mode v2_C
  python step07d_render_best.py --alpha 0.25 --mode v2_C
  python step07d_render_best.py --alpha 0.5  --mode v2_C
"""

import sys, os, argparse, gc, subprocess
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'

# ── nvdiffrast arch guard (copied from step07d) ───────────────────────────────
def _ensure_nvdiffrast():
    import subprocess, torch
    p        = torch.cuda.get_device_properties(0)
    arch_tag = f'sm_{p.major}{p.minor}'
    arch_str = f'{p.major}.{p.minor}'
    local    = f'/tmp/nvdiff_{arch_tag}'
    if local not in sys.path:
        sys.path.insert(0, local)
    try:
        import nvdiffrast.torch as _dr
        _dr.RasterizeCudaContext(); del _dr
        return
    except Exception:
        if os.environ.get('_NVDIFF_REBUILT') == arch_tag:
            raise RuntimeError(f'nvdiffrast still broken for {arch_tag} after rebuild')
    print(f'[NVDIFF] building for {arch_tag} -> {local} (~5 min)', flush=True)
    src = f'/tmp/nvdiff_src_{arch_tag}_{os.getpid()}'
    os.makedirs(src, exist_ok=True)
    pip = '/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/pip'
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': arch_str}
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/nvdiffrast.git',
                    f'{src}/nvdiffrast', '--depth', '1', '--quiet'], check=True, env=env)
    subprocess.run([pip, 'install', '.', '--target', local,
                    '--no-build-isolation', '--no-cache-dir', '--no-deps', '-q'],
                   cwd=f'{src}/nvdiffrast', env=env, check=True)
    print(f'[NVDIFF] build done — restarting', flush=True)
    os.environ['_NVDIFF_REBUILT'] = arch_tag
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_nvdiffrast()
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules   import sparse as sp

from step4_mcfm.mcfm            import mcfm_v2, mcfm_v3
from step6_5_lora.lora_v2_fixed import build_lora_blocks, freeze_trellis
from step6_5_lora.dual_path_v2  import dual_path_ctx_v2
from step8_decode_render.decode_render import (make_renderer, normalize_slat,
                                               decode_and_render,
                                               RENDER_RES, SLAT_MEAN, SLAT_STD)

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                        '/outputs/teapot_lava_kling_premium'
                        '/teapot_lava_kling_premium_front/all_frames_150')
GT_FRAME_75  = VIDEO_FRAMES_DIR / 'frame_0075.png'
PRETRAINED   = 'JeffreyXiang/TRELLIS-image-large'
DEVICE       = torch.device('cuda')
N_FRAMES     = 150
STRUCT_SEED  = 42
FIXED_SEED   = 6
STEPS        = 25
RESCALE_T    = 3.0

_t_seq  = np.linspace(1, 0, STEPS + 1)
_t_seq  = RESCALE_T * _t_seq / (1 + (RESCALE_T - 1) * _t_seq)
T_PAIRS = [(_t_seq[i], _t_seq[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_window(frame_idx, mode):
    if mode.endswith('_C'):
        return [frame_idx, min(N_FRAMES, frame_idx + 1)]
    return [max(1, frame_idx - 1), frame_idx, min(N_FRAMES, frame_idx + 1)]


def encode_frame(dino_model, frame_idx):
    img = Image.open(VIDEO_FRAMES_DIR / f'frame_{frame_idx:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    x   = _DINO_NORM(torch.from_numpy(arr).permute(2, 0, 1).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = dino_model(x, is_training=True)['x_prenorm']
        toks  = F.layer_norm(feats, feats.shape[-1:])
    return toks.squeeze(0)


def blend(mode, raw_tokens, frame_idx):
    win     = get_window(frame_idx, mode)
    lam     = torch.tensor([1.0 / len(win)] * len(win), dtype=torch.float32, device=DEVICE)
    tok_d   = {i: {'tokens': raw_tokens[i].to(DEVICE)} for i in set(win)}
    mcfm_fn = mcfm_v2 if mode.startswith('v2') else mcfm_v3
    K_hat, _ = mcfm_fn(tok_d, win, frame_idx, lam)
    stacked  = torch.stack([raw_tokens[i].to(DEVICE) for i in win], 0)
    weights  = torch.tensor([2.0 if i == frame_idx else 1.0 for i in win],
                             dtype=torch.float32, device=DEVICE)
    weights  /= weights.sum()
    K_pooled = (weights[:, None, None] * stacked).sum(0)
    return K_hat, K_pooled


def render_frame(flow_model, pipeline, lora_blocks, alpha,
                 raw_tokens, mode, frame_idx, coords, fixed_noise_feats, renderer):
    K_hat, K_pooled = blend(mode, raw_tokens, frame_idx)
    ns = sp.SparseTensor(feats=fixed_noise_feats.clone(), coords=coords)
    cond_gl = K_hat.unsqueeze(0)
    flow_model.to(DEVICE)
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            with dual_path_ctx_v2(flow_model, K_pooled.to(DEVICE), lora_blocks,
                                   alpha, enhance_bias=None):
                v = flow_model(ns, t_ten, cond_gl)
            ns = ns.replace(ns.feats - (t - t_prev) * v.feats)
    slat  = normalize_slat(ns)
    color, _ = decode_and_render(pipeline, slat, renderer, diag=False, device=DEVICE)
    return color.detach().clamp(0, 1)


def make_comp_strip(gt_path, ctrl_tensor, lora_tensor, label, font):
    gt_img   = Image.open(gt_path).convert('RGB').resize((RENDER_RES, RENDER_RES), Image.LANCZOS)
    ctrl_arr = (ctrl_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    lora_arr = (lora_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    ctrl_img = Image.fromarray(ctrl_arr)
    lora_img = Image.fromarray(lora_arr)

    bar_h = 22
    W, H  = RENDER_RES, RENDER_RES
    strip = Image.new('RGB', (W * 3, H + bar_h), (30, 30, 30))
    draw  = ImageDraw.Draw(strip)

    for col, (img, title) in enumerate([(gt_img, 'GT video'),
                                         (ctrl_img, 'MCFM ctrl (α=0)'),
                                         (lora_img, label)]):
        strip.paste(img, (col * W, bar_h))
        draw.text((col * W + 4, 3), title, fill=(220, 220, 220), font=font)

    return strip


def make_video(frame_dir, out_path, fps=10):
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', str(frame_dir / 'frame_*.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [WARN] ffmpeg failed: {result.stderr[:200]}')
    else:
        print(f'  [VIDEO] {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--mode',  type=str,   default='v2_C',
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    parser.add_argument('--lr',    type=float, default=1e-4)
    parser.add_argument('--rank',  type=int,   default=4)
    args = parser.parse_args()

    ALPHA = float(args.alpha)
    MODE  = args.mode
    LR    = args.lr
    RANK  = args.rank

    alpha_str = f'{ALPHA:.2f}'.replace('.', 'p')
    lr_tag    = ('' if abs(LR - 1e-4) < 1e-10
                 else '_lr' + f'{LR:.0e}'.replace('-0', 'm').replace('+0', ''))
    results_dir = _HERE / f'results_mcfm_{MODE}_d_alpha{alpha_str}{lr_tag}_seed6'
    ckpt_path   = results_dir / 'lora_ckpts' / 'lora_best.pt'

    print('=' * 72)
    print(f'step07d_render_best  mode={MODE}  alpha={ALPHA}  lr={LR}')
    print(f'  results : {results_dir}')
    print(f'  ckpt    : {ckpt_path}')
    print('=' * 72)

    if not ckpt_path.exists():
        print(f'[SKIP] No checkpoint at {ckpt_path}')
        return

    out_dir  = results_dir / 'full_render'
    ctrl_dir = out_dir / 'ctrl'
    lora_dir = out_dir / 'lora'
    comp_dir = out_dir / 'comp'
    for d in [ctrl_dir, lora_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = ImageFont.load_default()

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print('\n[LOAD] Loading pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']

    # ── Sparse structure ──────────────────────────────────────────────────────
    print(f'[STRUCT] seed={STRUCT_SEED}...')
    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    assert coords.shape[0] == 7301, f'N_vox={coords.shape[0]} != 7301'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    # unload unused models
    for name in list(pipeline.models.keys()):
        if name not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── LoRA blocks + load checkpoint ────────────────────────────────────────
    print(f'[LORA] Loading checkpoint: {ckpt_path.name}')
    freeze_trellis(flow_model)
    lora_blocks = build_lora_blocks(rank=RANK, lora_alpha=RANK).to(DEVICE)
    ckpt        = torch.load(ckpt_path, map_location=DEVICE)
    assert ckpt.get('script') == 'step07d', f"Wrong script: {ckpt.get('script')}"
    assert abs(ckpt.get('alpha', -1) - ALPHA) < 1e-6, \
        f"Checkpoint alpha={ckpt.get('alpha')} != --alpha={ALPHA}"
    lora_blocks.load_state_dict(ckpt['lora_state'])
    lora_blocks.eval()
    print(f'  loaded epoch={ckpt["epoch"]}  task_loss={ckpt["avg_task_loss"]:.5f}')

    # ── Fixed noise ───────────────────────────────────────────────────────────
    print(f'[NOISE] seed={FIXED_SEED}')
    torch.manual_seed(FIXED_SEED)
    fixed_noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, device=DEVICE)

    # ── DINOv2 encode all frames ──────────────────────────────────────────────
    print(f'[DINO] Encoding {N_FRAMES} frames...')
    dino_model = pipeline.models['image_cond_model'].to(DEVICE)
    raw_tokens = {}
    for i in range(1, N_FRAMES + 1):
        raw_tokens[i] = encode_frame(dino_model, i).cpu()
        if i % 50 == 0:
            print(f'  {i}/{N_FRAMES}')
    dino_model.cpu(); torch.cuda.empty_cache()

    renderer = make_renderer()
    pipeline.models['slat_decoder_mesh'].to(DEVICE)
    flow_model.eval()

    lora_label = f'LoRA α={ALPHA:.2f} e{ckpt["epoch"]:02d}'

    # ── Render all 150 frames ─────────────────────────────────────────────────
    print(f'\n[RENDER] {N_FRAMES} frames  ctrl + lora  ...')
    import time
    t0 = time.time()

    ctrl_cache = {}  # cache ctrl renders (reuse if already done)

    for fi in range(1, N_FRAMES + 1):
        lora_out = lora_dir / f'frame_{fi:04d}.png'
        ctrl_out = ctrl_dir / f'frame_{fi:04d}.png'
        comp_out = comp_dir / f'frame_{fi:04d}.png'

        # skip if all three already exist
        if lora_out.exists() and ctrl_out.exists() and comp_out.exists():
            print(f'  f{fi:03d} [skip — already rendered]')
            continue

        gt_path = VIDEO_FRAMES_DIR / f'frame_{fi:04d}.png'

        # ctrl render
        if ctrl_out.exists():
            ctrl_t = torch.from_numpy(
                np.array(Image.open(ctrl_out).convert('RGB')).astype(np.float32) / 255.0
            ).permute(2, 0, 1).to(DEVICE)
        else:
            ctrl_t = render_frame(flow_model, pipeline, lora_blocks, 0.0,
                                  raw_tokens, MODE, fi, coords, fixed_noise_feats, renderer)
            arr = (ctrl_t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(ctrl_out)

        # lora render
        if lora_out.exists():
            lora_t = torch.from_numpy(
                np.array(Image.open(lora_out).convert('RGB')).astype(np.float32) / 255.0
            ).permute(2, 0, 1).to(DEVICE)
        else:
            lora_t = render_frame(flow_model, pipeline, lora_blocks, ALPHA,
                                  raw_tokens, MODE, fi, coords, fixed_noise_feats, renderer)
            arr = (lora_t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(lora_out)

        # comp strip
        strip = make_comp_strip(gt_path, ctrl_t, lora_t, lora_label, font)
        strip.save(comp_out)

        elapsed = time.time() - t0
        eta     = elapsed / fi * (N_FRAMES - fi)
        print(f'  f{fi:03d}/{N_FRAMES}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')

        del ctrl_t, lora_t
        gc.collect(); torch.cuda.empty_cache()

    # ── Videos ───────────────────────────────────────────────────────────────
    print('\n[VIDEO] Encoding comparison video...')
    make_video(comp_dir, out_dir / 'comparison.mp4', fps=10)
    make_video(lora_dir, out_dir / 'lora_only.mp4',  fps=10)
    make_video(ctrl_dir, out_dir / 'ctrl_only.mp4',  fps=10)

    print(f'\n[DONE] Results in {out_dir}')
    print(f'  comp frames : {comp_dir}')
    print(f'  video       : {out_dir}/comparison.mp4')


if __name__ == '__main__':
    main()
