"""
measure_logits.py
-----------------
THE decisive measurement. Reads out_layer's raw colour logits directly.

Every claim about the white patches so far has been inferred by inverting
sigmoid on rendered pixels. This reads the actual tensor.

  latent -> frozen decoder -> out_layer[53:101] = colour LOGITS -> sigmoid -> RGB

  flexicubes.py:94   voxelgrid_colors = torch.sigmoid(voxelgrid_colors)
  cube2mesh.py:76-84 layout: sdf 0:8 | deform 8:32 | weights 32:53 | color 53:101
                     color is (8 corners, 6ch) = RGB + normal per corner

  sigmoid(2)=0.88  sigmoid(3)=0.95  sigmoid(4)=0.98  sigmoid(6)=0.9975

If rung16 reaches 4-6 where frozen sits at +-2, the gain hypothesis holds and the
fix is a drift penalty. If both sit in the same range, the hypothesis is wrong.

Original header follows.

The rendered PNG cannot answer this. render_mesh does

    color * mask + (1 - mask)

so the background is white BY CONSTRUCTION, and the frame is 8-bit, so anything
above 1.0 has already been clipped to 255 by the time it reaches disk. From the
image alone, "a hole in the mesh" and "colour pushed past 1.0" look identical.

So this measures the VERTEX COLOURS THEMSELVES -- mesh.vertex_attrs[:, :3], the
albedo the renderer interpolates -- before any rasterisation, clamping or
quantisation. Frozen vs rung15, same frames, same seeds.

If rung15's albedo exceeds 1.0 where frozen's does not, the white patches are
saturation caused by the adapter, and the magnitude of the overshoot is
recoverable here and nowhere else.

This is angle-independent: it is a property of the mesh, not of the viewpoint,
so it is a separate axis of evidence from the orbit sweep.

Usage:
  python .../measure_color_overshoot.py --frames 1 30 60 75 90 120 150
"""

import sys, os, argparse, json, math, gc
from pathlib import Path

os.environ['SPCONV_ALGO']           = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_HERE = Path(__file__).resolve().parent
_LEX  = _HERE.parent
_ROOT = _LEX.parent.parent
_PIPE = _ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, nargs='+', default=[1, 30, 60, 75, 90, 120, 150])
ap.add_argument('--run', default='rung16e1_rembg_uniform_all_qkvo_r4_s6_0f8620e0')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'logits')
OUT.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, p): self._f = open(p, 'a', buffering=1)
    def write(self, m): sys.__stdout__.write(m); self._f.write(m)
    def flush(self): sys.__stdout__.flush(); self._f.flush()


sys.stdout = _Tee(OUT / 'log.txt'); sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from contextlib import contextmanager

sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_PIPE))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.modules.sparse.attention import sparse_scaled_dot_product_attention
from step8_decode_render.decode_render import make_renderer, normalize_slat

DEVICE = torch.device('cuda')
GT_FRAMES_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                     '/outputs/teapot_lava_kling_premium'
                     '/teapot_lava_kling_premium_front/all_frames_150')
PRETRAINED, STRUCT_SEED, NOISE_SEED = 'JeffreyXiang/TRELLIS-image-large', 42, 6
STEPS, RESCALE_T = 25, 3.0
_ts = np.linspace(1, 0, STEPS + 1); _ts = RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)
T_PAIRS = [(_ts[i], _ts[i + 1]) for i in range(STEPS)]
_DINO_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class LoRALayer(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, i)); self.B = nn.Parameter(torch.zeros(o, r))
    def forward(self, x): return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class XAttnLoRABundle(nn.Module):
    def __init__(self, ca, rank, targets):
        super().__init__()
        self.targets = tuple(targets)
        for n in self.targets:
            lin = getattr(ca, n)
            setattr(self, f'lora_{n}', LoRALayer(lin.in_features, lin.out_features, rank))
    def get(self, n):
        return getattr(self, f'lora_{n}', None) if n in self.targets else None


class XAttnLoRARegistry(nn.Module):
    def __init__(self, fm, active, rank, targets):
        super().__init__()
        mods, i = {}, 0
        for blk in fm.blocks:
            if not hasattr(blk, 'cross_attn'): continue
            if i in set(active): mods[str(i)] = XAttnLoRABundle(blk.cross_attn, rank, targets)
            i += 1
        self.blocks = nn.ModuleDict(mods)
    def get(self, i):
        k = str(i); return self.blocks[k] if k in self.blocks else None


LORA = None


def _fwd(module, x, context, idx):
    """modules.py:126-139 plus rung15's three deltas. Nothing else changed."""
    lb = LORA.get(idx) if LORA is not None else None
    q_sp = module._linear(module.to_q, x)
    if lb is not None and lb.get('to_q') is not None:
        q_sp = q_sp.replace(q_sp.feats + lb.get('to_q')(x.feats).to(q_sp.feats.dtype))
    q = module._reshape_chs(q_sp, (module.num_heads, -1))
    kv_t = module._linear(module.to_kv, context)
    if lb is not None and lb.get('to_kv') is not None:
        kv_t = kv_t + lb.get('to_kv')(context).to(kv_t.dtype)
    kv = module._fused_pre(kv_t, num_fused=2)
    h = sparse_scaled_dot_product_attention(q, kv)
    h = module._reshape_chs(h, (-1,))
    out = module._linear(module.to_out, h)
    if lb is not None and lb.get('to_out') is not None:
        out = out.replace(out.feats + lb.get('to_out')(h.feats).to(out.feats.dtype))
    return out


@contextmanager
def lora_ctx(fm):
    saved, i = {}, 0
    for blk in fm.blocks:
        if not hasattr(blk, 'cross_attn'): continue
        ca = blk.cross_attn; saved[i] = ca.forward
        def _mk(m, idx):
            def _f(x, context=None): return _fwd(m, x, context, idx)
            return _f
        ca.forward = _mk(ca, i); i += 1
    assert i == 24
    try: yield
    finally:
        j = 0
        for blk in fm.blocks:
            if hasattr(blk, 'cross_attn') and j in saved:
                blk.cross_attn.forward = saved[j]; j += 1


def encode(dino, i):
    img = Image.open(GT_FRAMES_DIR / f'frame_{i:04d}.png').convert('RGB').resize((518, 518), Image.LANCZOS)
    a = np.array(img).astype(np.float32) / 255.0
    x = _DINO_NORM(torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(x, is_training=True)['x_prenorm']
        return F.layer_norm(f, f.shape[-1:]).squeeze(0)


def denoise(fm, noise, coords, cond):
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad(), lora_ctx(fm):
        for t, tp in T_PAIRS:
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            v = fm(x, tt, cond)
            x = x.replace(x.feats - (t - tp) * v.feats)
    return normalize_slat(x)



COLOR_START, COLOR_END = 53, 101      # cube2mesh.py LAYOUTS


def logit_stats(L):
    """L: [V, 24] RGB logits (8 corners x 3), pre-sigmoid."""
    f = L.flatten().float()
    q = lambda p: float(f.quantile(p))
    return dict(n=int(f.numel()), mean=float(f.mean()), std=float(f.std()),
                p01=q(0.01), p50=q(0.50), p99=q(0.99), p999=q(0.999),
                mn=float(f.min()), mx=float(f.max()),
                frac_gt3=float((f > 3).float().mean()),
                frac_gt4=float((f > 4).float().mean()),
                frac_gt6=float((f > 6).float().mean()),
                frac_gt8=float((f > 8).float().mean()))


def main():
    print('=' * 100)
    print('RAW COLOUR LOGITS at out_layer, pre-sigmoid.  frozen vs adapted.')
    print(f'  frames {args.frames}   run {args.run}')
    print('=' * 100, flush=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained(PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    assert coords.shape[0] == 7301
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass

    dino = pipe.models['image_cond_model'].to(DEVICE)
    toks = {f: encode(dino, f).cpu() for f in args.frames}
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(NOISE_SEED)
    noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)

    global LORA
    rd = _LEX / 'runs' / args.run
    cfg = json.load(open(rd / 'config.json'))
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu',
                    weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'],
                            tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    assert max(float(getattr(b, f'lora_{n}').B.float().norm())
               for b in reg.blocks.values() for n in b.targets) > 1e-6
    print(f'\n[LORA] {args.run}  epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}\n')

    cap = {}
    def _hook(m, i, o):
        cap['f'] = o.feats.detach()
    h = dec.out_layer.register_forward_hook(_hook)

    rows = []
    hdr = (f"{'frame':>5} {'model':>7} {'n_logits':>11} {'mean':>7} {'std':>6} "
           f"{'p01':>7} {'p50':>7} {'p99':>7} {'max':>7} | "
           f"{'>3':>7} {'>4':>7} {'>6':>7} {'>8':>7}")
    print(hdr); print('-' * len(hdr))
    try:
        for f in args.frames:
            cond = toks[f].unsqueeze(0).to(DEVICE)
            for name in ('frozen', 'adapted'):
                LORA = None if name == 'frozen' else reg
                slat = denoise(fm, noise, coords, cond)
                cap.clear()
                with torch.no_grad():
                    dec(slat)
                assert 'f' in cap, 'out_layer hook did not fire'
                col = cap['f'][:, COLOR_START:COLOR_END]        # [V, 48]
                assert col.shape[1] == 48, f'expected 48 colour ch, got {col.shape[1]}'
                rgb = col.reshape(col.shape[0], 8, 6)[..., :3].reshape(col.shape[0], 24)
                s = logit_stats(rgb); s.update(frame=f, model=name); rows.append(s)
                print(f"{f:>5} {name:>7} {s['n']:>11,} {s['mean']:>7.3f} {s['std']:>6.3f} "
                      f"{s['p01']:>7.3f} {s['p50']:>7.3f} {s['p99']:>7.3f} {s['mx']:>7.3f} | "
                      f"{100*s['frac_gt3']:>6.2f}% {100*s['frac_gt4']:>6.2f}% "
                      f"{100*s['frac_gt6']:>6.2f}% {100*s['frac_gt8']:>6.2f}%", flush=True)
                del slat, col, rgb
                gc.collect(); torch.cuda.empty_cache()
    finally:
        h.remove()

    json.dump(rows, open(OUT / 'logits.json', 'w'), indent=2)
    fz = [r for r in rows if r['model'] == 'frozen']
    ad = [r for r in rows if r['model'] == 'adapted']
    print('\n' + '=' * 100)
    print('VERDICT')
    print(f"  frozen   p99 {np.mean([r['p99'] for r in fz]):.3f}   max {np.mean([r['mx'] for r in fz]):.3f}   "
          f">4: {100*np.mean([r['frac_gt4'] for r in fz]):.3f}%")
    print(f"  adapted  p99 {np.mean([r['p99'] for r in ad]):.3f}   max {np.mean([r['mx'] for r in ad]):.3f}   "
          f">4: {100*np.mean([r['frac_gt4'] for r in ad]):.3f}%")
    d99 = np.mean([r['p99'] for r in ad]) - np.mean([r['p99'] for r in fz])
    print(f"\n  p99 logit shift = {d99:+.3f}")
    print(f"  rung13's clamp ceiling was 8.0 -> sigmoid 0.9997")
    if np.mean([r['frac_gt4'] for r in ad]) > 3 * max(np.mean([r['frac_gt4'] for r in fz]), 1e-6):
        print('  -> adapted drives many more logits past 4 (sigmoid 0.98). GAIN hypothesis SUPPORTED.')
    else:
        print('  -> adapted logits sit where frozen does. GAIN hypothesis REFUTED; look elsewhere.')
    print('=' * 100, flush=True)


if __name__ == '__main__':
    main()
