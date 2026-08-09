"""
measure_feature_separability.py
-------------------------------
Can a SHARED LoRA matrix distinguish supervised tokens from unsupervised ones?

WHY THIS DECIDES THE NEXT EXPERIMENT
  The proposed Vox-E-style regulariser adds  lambda * ||z_lora - z_frozen||  over
  every token, so the unsupervised ones finally contribute to dL/dB. But B is a
  single shared matrix and the ONLY thing it sees is a token's feature vector:

     features separable   -> some B edits front-like features and not back-like
                             ones; the regulariser can TARGET. Real fix.
     features inseparable -> no B can do both; the render gradient and the
                             regulariser gradient fight and the optimiser just
                             damps the whole edit. That reproduces the LoRA-scale
                             sweep, which already showed no uniform scale gives
                             correct brightness without whitening.

FEATURE SPACE -- the point of this script, and an earlier version got it wrong.
  B does NOT consume the 8-channel SLaT latent. It consumes the flow block's
  HIDDEN STATE at the cross-attention input. A probe on the 8-dim latent says
  nothing about what a shared B can separate, and its nearest-neighbour cosines
  are inflated by density -- in 8 dims with a few thousand points a near-duplicate
  always exists. So the hidden state is captured by forward hook and every metric
  runs on THAT.

GRIDS.  Three, and the hidden state lives on the last one:
      467,264  decoder fine cubes
        7,301  sparse-structure voxels @ 64^3   (the flow's INPUT)
        1,748  transformer tokens     @ 32^3    (patch_size=2 downsamples first)
  Token coords are read from the captured SparseTensor itself, not derived, so
  the labels cannot silently misalign with the features.

METRICS, per probed block
  1. VISIBILITY per token from the training camera at yaw=0. Rasterize the mesh
     for a depth buffer, project each token centre, compare depth. An occlusion
     test, not a normal test.
  2. AUC separating visible from occluded using the hidden state alone (logistic
     probe, held out). 0.5 = a shared linear map CANNOT tell them apart, so the
     regulariser can only damp. 0.9+ = separable, it can target. THE HEADLINE.
  3. occluded -> nearest visible token in feature space: cosine and the 3D
     distance to that partner. High cosine at large 3D distance is the mechanism
     by which unseen surface inherits the edit.
  4. Mirror-pair test: reflect each token through the object axis, report
     cosine(z_i, z_mirror). Direct "does the back look like the front" measure.
"""

import sys, os, argparse, math, gc
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LEX  = _HERE.parent
_ROOT = _LEX.parent.parent

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--frame', type=int, default=1)
_pre.add_argument('--run', default='rung19_intersection_uniform_all_qkvo_r4_s6_276b5db7')
_pre.add_argument('--probe-blocks', type=int, nargs='+', default=[0, 11, 23])
_pre.add_argument('--probe-knot', type=int, default=12)
A, _ = _pre.parse_known_args()

OUT = _HERE / 'separability'; OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_xattn_mesh.py', '--run', A.run, '--frame', str(A.frame),
            '--no-glb', '--out-dir', str(OUT / '_scratch')]
sys.path.insert(0, str(_ROOT / 'render'))
import export_xattn_mesh as X                                   # noqa: E402

import numpy as np, torch                                       # noqa: E402
from PIL import Image                                           # noqa: E402

sys.path.insert(0, str(_ROOT / 'experiments' / 'dynamic_texture_trellis_pipeline'))
import trellis.modules.sparse as sp                             # noqa: E402
from trellis.representations.mesh import MeshExtractResult      # noqa: E402
from step8_decode_render.decode_render import (                 # noqa: E402
    make_renderer, RENDER_RES, EXTRINSICS, INTRINSICS, normalize_slat)

DEVICE = X.DEVICE
FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))


def auc_of(scores, labels):
    o = scores.argsort(); yr = labels[o]
    npos, nneg = yr.sum(), (1 - yr).sum()
    if npos == 0 or nneg == 0: return float('nan')
    a = ((torch.cumsum(1 - yr, 0) * yr).sum() / (npos * nneg)).item()
    return max(a, 1 - a)


def probe(feat, y, seed=0):
    """Held-out logistic probe. Returns AUC of features -> visibility."""
    z = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g).to(DEVICE)
    ntr = int(0.7 * len(z)); tr, te = perm[:ntr], perm[ntr:]
    w = torch.zeros(z.shape[1], 1, device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(800):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            (z[tr] @ w).squeeze() + b, y[tr]).backward()
        opt.step()
    return auc_of(((z[te] @ w).squeeze() + b).detach(), y[te])


def main():
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained(X.PRETRAINED); pipe.to(DEVICE)
    fm, dec = pipe.models['slat_flow_model'], pipe.models['slat_decoder_mesh']
    for p in fm.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ref = Image.open(X.GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cs = pipe.get_cond([ref]); torch.manual_seed(X.STRUCT_SEED)
    coords = pipe.sample_sparse_structure(cs, num_samples=1)
    assert coords.shape[0] == 7301
    del cs; gc.collect(); torch.cuda.empty_cache()
    for n in list(pipe.models):
        if n not in {'slat_flow_model', 'slat_decoder_mesh', 'image_cond_model'}:
            try: pipe.models[n].cpu()
            except Exception: pass
    dino = pipe.models['image_cond_model'].to(DEVICE)
    cond = X.encode(dino, A.frame).unsqueeze(0).to(DEVICE)
    dino.cpu(); gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(X.NOISE_SEED)
    noise = torch.randn(coords.shape[0], fm.in_channels, device=DEVICE)

    # ── capture the cross-attention INPUT: features AND their own coords ──────
    CAP = {'knot': -1}

    def mk(idx):
        def h(mod, inp, out):
            if CAP['knot'] == A.probe_knot and idx in A.probe_blocks:
                CAP[idx] = (inp[0].feats.detach().float().clone(),
                            inp[0].coords[:, 1:].detach().clone())
        return h

    hs, ci = [], 0
    for blk in fm.blocks:
        if not hasattr(blk, 'cross_attn'): continue
        hs.append(blk.cross_attn.register_forward_hook(mk(ci))); ci += 1
    assert ci == 24, ci

    X.LORA = None
    x = sp.SparseTensor(feats=noise.clone(), coords=coords)
    with torch.no_grad():
        for k, (t, tp) in enumerate(X.T_PAIRS):
            CAP['knot'] = k
            tt = torch.tensor([1000.0 * t], device=DEVICE, dtype=torch.float32)
            x = x.replace(x.feats - (t - tp) * fm(x, tt, cond).feats)
    for h in hs: h.remove()
    missing = [b for b in A.probe_blocks if b not in CAP]
    assert not missing, f'capture failed for blocks {missing}'
    slat = normalize_slat(x)
    with torch.no_grad(): mesh = dec(slat)[0]

    ntok = CAP[A.probe_blocks[0]][0].shape[0]
    print(f'[CAPTURE] knot {A.probe_knot}: ' +
          '  '.join(f'blk{b} {tuple(CAP[b][0].shape)}' for b in A.probe_blocks))
    assert ntok == 1748, f'expected 1,748 tokens @32^3, got {ntok}'

    # ── visibility of each TOKEN centre from the training camera ─────────────
    tc = CAP[A.probe_blocks[0]][1].float()          # coords on the 32^3 grid
    P = (tc + 0.5) / 32.0 - 0.5
    Pw = X.align(P)

    r = make_renderer(DEVICE)
    v = X.align(mesh.vertices.detach()); f = mesh.faces.detach()
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    f = f[0.5 * torch.cross(e1, e2, dim=1).norm(dim=1) > 1e-6]
    col = mesh.vertex_attrs[:, :3].detach().clamp(0, 1)
    res = r.render(MeshExtractResult(vertices=v, faces=f,
                   vertex_attrs=torch.cat([col, col], 1), res=256),
                   EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE),
                   return_types=['mask', 'depth'])
    depth = res['depth'].squeeze(); mask = res['mask'].squeeze() > 0.5

    homo = torch.cat([Pw, torch.ones_like(Pw[:, :1])], 1)
    cam = homo @ EXTRINSICS.to(DEVICE).T
    zc = cam[:, 2]
    px = ((FX_N * cam[:, 0] / zc.clamp(min=1e-8)) + 0.5) * RENDER_RES
    py = ((FX_N * cam[:, 1] / zc.clamp(min=1e-8)) + 0.5) * RENDER_RES
    ix = px.round().long().clamp(0, RENDER_RES - 1)
    iy = py.round().long().clamp(0, RENDER_RES - 1)
    # tolerance ~1 token voxel (1/32 world units) so surface tokens aren't
    # rejected by their own thickness
    VIS = mask[iy, ix] & (zc <= depth[iy, ix] + (1.0 / 32.0))
    y = VIS.float()
    print(f'\ntokens {ntok:,}   visible at yaw=0 {int(VIS.sum()):,} '
          f'({100 * float(y.mean()):.1f}%)   occluded {int((~VIS).sum()):,}')

    print(f'\n{"blk":>4s} {"dim":>5s} {"AUC":>7s} | {"occ->vis cos":>12s} '
          f'{"3D dist":>8s} {">0.99":>7s} | {"mirror cos":>10s}')
    print('-' * 72)
    rows = {}
    for b in A.probe_blocks:
        Z = CAP[b][0]
        a = probe(Z, y)
        zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)
        zv, zo = zn[VIS], zn[~VIS]
        pv, po = Pw[VIS], Pw[~VIS]
        best, bi = (zo @ zv.T).max(dim=1)
        d3 = (po - pv[bi]).norm(dim=1)
        Pm = Pw.clone(); Pm[:, 1] = 2 * Pw[:, 1].mean() - Pw[:, 1]
        dm = torch.cdist(Pm, Pw); mi = dm.argmin(dim=1); ok = dm.min(dim=1).values < 0.05
        cm = (zn[ok] * zn[mi[ok]]).sum(1)
        print(f'{b:4d} {Z.shape[1]:5d} {a:7.4f} | {best.mean():12.4f} '
              f'{d3.mean():8.4f} {100*float((best>0.99).float().mean()):6.1f}% | '
              f'{cm.mean():10.4f}')
        rows[b] = dict(auc=a, cos=float(best.mean()), d3=float(d3.mean()),
                       cos99=float((best > 0.99).float().mean()), mirror=float(cm.mean()))

    print('\n  AUC 0.50 -> shared B cannot target; the regulariser can only DAMP')
    print('  AUC 0.90 -> shared B can target; the regulariser is a real fix')
    np.savez_compressed(OUT / 'sep_hidden.npz',
                        vis=VIS.cpu().numpy(),
                        **{f'blk{b}_{k}': v for b, d in rows.items() for k, v in d.items()})
    print(f'\n[DONE] {OUT}/sep_hidden.npz')


if __name__ == '__main__':
    main()
