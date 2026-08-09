"""
measure_color_overshoot.py
--------------------------
Why are there white patches on the rung15 renders?

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
ap.add_argument('--run', default='rung15v1_uniform_all_qkvo_r4_s6_2674ec70')
ap.add_argument('--out', default=None)
args = ap.parse_args()

OUT = Path(args.out) if args.out else (_HERE / 'color_overshoot')
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


def stats(c):
    """c: [V,3] albedo, as the renderer receives it."""
    f = c.flatten()
    return {
        'n_vert': int(c.shape[0]),
        'frac_over_1': float((c > 1.0).any(dim=1).float().mean()),
        'frac_chan_over_1': float((f > 1.0).float().mean()),
        'frac_over_099': float((c > 0.99).any(dim=1).float().mean()),
        'max': float(f.max()), 'p999': float(f.quantile(0.999)),
        'p99': float(f.quantile(0.99)), 'mean': float(f.mean()),
        'frac_under_0': float((f < 0.0).float().mean()),
        'min': float(f.min()),
    }


def main():
    print('=' * 96)
    print('vertex-colour overshoot: are the white patches CLIPPING, before any render?')
    print(f'  frames {args.frames}   run {args.run}')
    print('=' * 96, flush=True)

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
    ck = torch.load(rd / 'lora_ckpts' / 'lora_best.pt', map_location='cpu', weights_only=True)
    reg = XAttnLoRARegistry(fm, cfg['active_blocks'], cfg['rank'], tuple(cfg['targets'])).to(DEVICE)
    reg.load_state_dict(ck['registry_state'], strict=True); reg.eval()
    bn = max(float(getattr(b, f'lora_{n}').B.float().norm())
             for b in reg.blocks.values() for n in b.targets)
    assert bn > 1e-6, 'LoRA B all zero — adapter not loaded'
    print(f'\n[LORA] epoch {ck["epoch"]}  psnr {ck["best_psnr"]:.3f}  max||B|| {bn:.4f}\n')

    rows = []
    hdr = (f"{'frame':>5} {'model':>7} {'verts':>9} {'>1.0':>8} {'>0.99':>8} "
           f"{'max':>7} {'p99.9':>7} {'p99':>7} {'mean':>7} {'<0':>7} {'min':>8}")
    print(hdr); print('-' * len(hdr))
    for f in args.frames:
        cond = toks[f].unsqueeze(0).to(DEVICE)
        for name in ('frozen', 'rung15'):
            LORA = None if name == 'frozen' else reg
            slat = denoise(fm, noise, coords, cond)
            with torch.no_grad():
                mesh = dec(slat)[0]
            c = mesh.vertex_attrs[:, :3].detach().float()
            s = stats(c); s.update(frame=f, model=name); rows.append(s)
            print(f"{f:>5} {name:>7} {s['n_vert']:>9,} {100*s['frac_over_1']:>7.2f}% "
                  f"{100*s['frac_over_099']:>7.2f}% {s['max']:>7.3f} {s['p999']:>7.3f} "
                  f"{s['p99']:>7.3f} {s['mean']:>7.3f} {100*s['frac_under_0']:>6.2f}% "
                  f"{s['min']:>8.3f}", flush=True)
            del slat, mesh, c
            gc.collect(); torch.cuda.empty_cache()

    json.dump(rows, open(OUT / 'overshoot.json', 'w'), indent=2)
    print('\n' + '=' * 96)
    print('READING IT')
    print('  >1.0    fraction of VERTICES with any albedo channel above 1.0.')
    print('          These clip to white on render. The renderer cannot show how far above.')
    print('  <0      fraction of channels below 0, which clip to black.')
    print(f'\n  wrote {OUT}/overshoot.json')
    print('=' * 96, flush=True)


if __name__ == '__main__':
    main()
