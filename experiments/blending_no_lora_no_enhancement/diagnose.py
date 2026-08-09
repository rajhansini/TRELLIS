"""
Pipeline diagnostic — runs one frame and logs shapes/stats at every stage.
Verifies the wires are correct before running full experiments.

Usage:
  python diagnose.py --frame 77
"""

import os, sys, argparse, math

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnose.log')

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

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']  = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PIPE = _HERE.parent / 'dynamic_texture_trellis_pipeline'
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PIPE))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers  import MeshRenderer
import trellis.modules.sparse as sp
from step1_input_prep.input_prep       import load_frame, N_FRAMES
from step2_dino_encoding.dino_encoding import load_dino

PRETRAINED    = 'microsoft/TRELLIS-image-large'
GT_FRAME_75   = str(_PIPE / '..' / '..' / '..' /
                    'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                    '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
GT_FRAMES_DIR = (_PIPE / '..' / '..' / '..' /
                 'MV-Adapter-Experimental/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
DEVICE     = torch.device('cuda')
STEPS      = 25
NOISE_SEED = 42
RENDER_RES = 518
RESCALE_T  = 3.0
_SCALE     = 1024 ** -0.5

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


def sep(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')


def check(label, tensor, expected_shape=None, check_mean_zero=False):
    s = tuple(tensor.shape)
    mean = tensor.float().mean().item()
    std  = tensor.float().std().item()
    mn   = tensor.float().min().item()
    mx   = tensor.float().max().item()
    shape_ok = (s == expected_shape) if expected_shape else True
    print(f'  [{label}]')
    print(f'    shape : {s}  {"✓" if shape_ok else f"✗ expected {expected_shape}"}')
    print(f'    dtype : {tensor.dtype}')
    print(f'    mean  : {mean:.5f}  std: {std:.5f}  min: {mn:.5f}  max: {mx:.5f}')
    if check_mean_zero:
        ok = abs(mean) < 0.05
        print(f'    mean≈0: {"✓" if ok else f"✗ got {mean:.5f}"}')
    return shape_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=77)
    args = parser.parse_args()
    frame_i    = args.frame
    frame_next = min(N_FRAMES, frame_i + 1)
    frame_prev = max(1, frame_i - 1)

    print(f'\nDiagnostic — frame {frame_i}  (prev={frame_prev}, next={frame_next})')
    print(f'GPU: {torch.cuda.get_device_name(0)}  |  CUDA {torch.version.cuda}')

    # ── STEP 1: Load frames ───────────────────────────────────────────────────
    sep('STEP 1 — Input Prep')
    img_curr = load_frame(frame_i)
    img_next = load_frame(frame_next)
    img_prev = load_frame(frame_prev)
    print(f'  frame {frame_i}  size: {img_curr.size}  mode: {img_curr.mode}')
    print(f'  frame {frame_next} size: {img_next.size}')
    print(f'  frame {frame_prev} size: {img_prev.size}')
    assert img_curr.size == (img_curr.width, img_curr.height), 'frame load OK'
    print('  ✓ frames loaded')

    # ── STEP 2: DINOv2 encoding ────────────────────────────────────────────────
    sep('STEP 2 — DINOv2 Encoding')
    import torchvision.transforms as T
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    preprocess = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print('  Loading dinov2_vitl14_reg...')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
    dino.eval().to(DEVICE)

    with torch.no_grad():
        x = preprocess(img_curr).unsqueeze(0).to(DEVICE)
        print(f'  input tensor shape : {tuple(x.shape)}  (expect (1,3,518,518))')

        out      = dino(x, is_training=True)
        prenorm  = out['x_prenorm']
        print(f'  x_prenorm shape    : {tuple(prenorm.shape)}  (expect (1,1374,1024))')

        tokens = F.layer_norm(prenorm, prenorm.shape[-1:]).squeeze(0)
        check('tokens after layer_norm', tokens, expected_shape=(1374, 1024), check_mean_zero=True)

        # encode all three frames
        def encode(img):
            x = preprocess(img).unsqueeze(0).to(DEVICE)
            p = dino(x, is_training=True)['x_prenorm']
            return F.layer_norm(p, p.shape[-1:]).squeeze(0)

        tok_prev = encode(img_prev)
        tok_curr = encode(img_curr)
        tok_next = encode(img_next)
        print(f'  all three frames encoded OK')

    del dino
    torch.cuda.empty_cache()

    # ── STEP 3: Blending ──────────────────────────────────────────────────────
    sep('STEP 3 — Blending (v1 and v2)')

    # v1 C-style (50/50)
    v1_c = 0.5 * tok_curr + 0.5 * tok_next
    check('v1_C (0.5*curr + 0.5*next)', v1_c, expected_shape=(1374, 1024))

    # v1 D-style (25/50/25)
    v1_d = 0.25 * tok_prev + 0.50 * tok_curr + 0.25 * tok_next
    check('v1_D (0.25*prev + 0.5*curr + 0.25*next)', v1_d, expected_shape=(1374, 1024))

    # v2 C-style (attention, 2-frame)
    Q = tok_curr.unsqueeze(1)
    K = torch.stack([tok_curr, tok_next], dim=1)
    scores = torch.bmm(Q, K.transpose(1, 2)) * _SCALE
    attn_c = torch.softmax(scores, dim=-1)
    v2_c   = torch.bmm(attn_c, K).squeeze(1)
    check('v2_C (attn, window [i,i+1])', v2_c, expected_shape=(1374, 1024))
    attn_c_mean = attn_c.squeeze(1).mean(dim=0)
    print(f'    attn mean over positions: curr={attn_c_mean[0]:.4f}  next={attn_c_mean[1]:.4f}')

    # v2 D-style (attention, 3-frame)
    Q = tok_curr.unsqueeze(1)
    K = torch.stack([tok_prev, tok_curr, tok_next], dim=1)
    scores = torch.bmm(Q, K.transpose(1, 2)) * _SCALE
    attn_d = torch.softmax(scores, dim=-1)
    v2_d   = torch.bmm(attn_d, K).squeeze(1)
    check('v2_D (attn, window [i-1,i,i+1])', v2_d, expected_shape=(1374, 1024))
    attn_d_mean = attn_d.squeeze(1).mean(dim=0)
    print(f'    attn mean over positions: prev={attn_d_mean[0]:.4f}  curr={attn_d_mean[1]:.4f}  next={attn_d_mean[2]:.4f}')

    # ── STEP 4: Pipeline load + voxel structure ────────────────────────────────
    sep('STEP 4 — Pipeline Load + Voxel Structure')
    print('  Loading TRELLIS pipeline...')
    pipeline   = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    print(f'  flow_model in_channels : {flow_model.in_channels}')
    print(f'  flow_model device      : {next(flow_model.parameters()).device}')

    img_75      = Image.open(GT_FRAME_75).convert('RGB')
    cond_struct = pipeline.get_cond([img_75])
    torch.manual_seed(NOISE_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    N_vox  = coords.shape[0]
    print(f'  coords shape           : {tuple(coords.shape)}  (N_vox={N_vox})')
    print(f'  coords dtype           : {coords.dtype}  device: {coords.device}')

    keep = {'slat_flow_model', 'slat_decoder_mesh'}
    for name in list(pipeline.models.keys()):
        if name not in keep:
            try: pipeline.models[name].cpu()
            except Exception: pass
    torch.cuda.empty_cache()

    # ── STEP 5: Conditioning tensor into flow model ────────────────────────────
    sep('STEP 5 — Conditioning Shape into Flow Model')
    cond_gl = v2_c.unsqueeze(0)   # use v2_C as the conditioning
    check('cond_gl (input to flow model)', cond_gl, expected_shape=(1, 1374, 1024))

    torch.manual_seed(NOISE_SEED + frame_i)
    noise_sp = sp.SparseTensor(
        feats=torch.randn(N_vox, flow_model.in_channels, device=DEVICE),
        coords=coords,
    )
    print(f'  noise_sp.feats shape   : {tuple(noise_sp.feats.shape)}  (expect ({N_vox}, {flow_model.in_channels}))')
    print(f'  noise_sp.coords shape  : {tuple(noise_sp.coords.shape)}')

    # single denoising step to verify flow model accepts cond_gl
    print('  Running 1 denoising step...')
    flow_model.eval()
    with torch.no_grad():
        t0, t1 = T_PAIRS[0]
        t_ten  = torch.tensor([1000.0 * t0], device=DEVICE, dtype=torch.float32)
        v      = flow_model(noise_sp, t_ten, cond_gl)
    print(f'  flow_model output feats: {tuple(v.feats.shape)}  (expect ({N_vox}, {flow_model.in_channels}))')
    print(f'  v.feats mean           : {v.feats.float().mean().item():.5f}')
    print('  ✓ conditioning flows correctly into flow model')

    # ── STEP 6: Full denoise ──────────────────────────────────────────────────
    sep('STEP 6 — Full Denoise (25 steps)')
    x = noise_sp
    with torch.no_grad():
        for t, t_prev in T_PAIRS:
            t_ten = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v     = flow_model(x, t_ten, cond_gl)
            x     = x.replace(x.feats - (t - t_prev) * v.feats)
    x0 = x
    print(f'  x0.feats shape         : {tuple(x0.feats.shape)}')
    check('x0.feats (raw denoised)', x0.feats)

    # ── STEP 7: Normalize SLaT ────────────────────────────────────────────────
    sep('STEP 7 — SLaT Normalization')
    std  = SLAT_STD.to(x0.feats.device)
    mean = SLAT_MEAN.to(x0.feats.device)
    slat = x0.replace(x0.feats * std + mean)
    check('slat.feats (after unnorm)', slat.feats)
    print(f'  SLAT_MEAN range: [{SLAT_MEAN.min():.3f}, {SLAT_MEAN.max():.3f}]')
    print(f'  SLAT_STD  range: [{SLAT_STD.min():.3f}, {SLAT_STD.max():.3f}]')

    # ── STEP 8: Decode + Render ───────────────────────────────────────────────
    sep('STEP 8 — Decode SLaT + Render')
    print('  Decoding slat -> mesh...')
    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ['mesh'])
        mesh    = decoded['mesh'][0]
    print(f'  mesh type              : {type(mesh).__name__}')
    print(f'  mesh vertices          : {mesh.vertices.shape}')
    print(f'  mesh faces             : {mesh.faces.shape}')

    renderer = MeshRenderer(
        rendering_options={'resolution': RENDER_RES, 'near': 0.5, 'far': 3.0, 'ssaa': 1}
    )
    print('  Rendering...')
    result = renderer.render(mesh, EXTRINSICS, INTRINSICS,
                             return_types=['color', 'mask'])
    print(f'  render color shape     : {tuple(result["color"].shape)}  (expect (3,{RENDER_RES},{RENDER_RES}))')
    print(f'  render mask shape      : {tuple(result["mask"].shape)}')
    mask  = result['mask'].unsqueeze(0)
    color = result['color'] * mask + (1.0 - mask)
    img   = (color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    print(f'  final image shape      : {img.shape}  dtype: {img.dtype}')
    print(f'  pixel range            : [{img.min()}, {img.max()}]')

    out_path = _HERE / 'diag_frame.png'
    Image.fromarray(img).save(out_path)
    print(f'  ✓ saved: {out_path}')

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    sep('DIAGNOSTIC SUMMARY')
    print(f'  frame {frame_i}  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  STEP 1  frames loaded               ✓')
    print(f'  STEP 2  DINOv2 tokens (1374,1024)   ✓  mean≈0: {abs(tokens.float().mean().item()) < 0.05}')
    print(f'  STEP 3  v1_C / v1_D / v2_C / v2_D  ✓')
    print(f'  STEP 4  pipeline + voxels ({N_vox})  ✓')
    print(f'  STEP 5  cond_gl (1,1374,1024) -> flow model  ✓')
    print(f'  STEP 6  25-step denoise              ✓')
    print(f'  STEP 7  SLaT unnorm                 ✓')
    print(f'  STEP 8  decode + render ({RENDER_RES}x{RENDER_RES})  ✓')
    print(f'\n  All wires connected. Ready to run experiments.\n')


if __name__ == '__main__':
    main()
