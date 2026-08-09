"""
render_rung14v1_video.py — the videos for the rung14-v1 cross-attention arm

rung14_v1_crossattn_lora.py trains and evaluates, but only ever wrote per-epoch
PNG strips. This renders the two videos that actually show the result:

  --sweep time    frame 1..150, CAMERA FIXED     -> the dynamic texture
  --sweep angle   frame PINNED, camera orbits    -> the 3D-propagation test
  --sweep both    frame advances AND orbits      -> one turn across 150 frames

The second one is the point. rung13 put the adapter in the mesh decoder, which
sits AFTER all 3D reasoning, and measured the consequence: the training camera
sees 16.8% of vertices, the adapter edits 100% of them at equal strength (ratio
1.02), and R^2 of the edit against the camera's image axis is 0.16-0.23 versus
~0.00 for the frozen colour. It learned a view-space projection, so it degrades
under rotation. rung14-v1 moves the adapter to cross-attention, upstream of 24
blocks of FROZEN self-attention, betting that TRELLIS's own propagation carries
the edit to surface the camera never saw. Only a turntable can show that.

Nothing here is re-derived: the flow/decode/render path, the alignment, the
camera and the LoRA context are all imported from the training script, so what
is rendered is exactly what was trained.
"""

import os, sys, json, gc, math, subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DEFAULT = 'rung14v1_all_qkvo_r4_s6_286f22f0'

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--run', default=_RUN_DEFAULT)
_ap.add_argument('--ckpt', default='lora_best.pt')
_ap.add_argument('--sweep', default='time', choices=['time', 'angle', 'both'],
                 help="time  = 150 frames, camera FIXED (the dynamic texture); "
                      "angle = frame pinned, camera orbits 360 (the propagation "
                      "test — shape and texture constant, so anything that "
                      "changes is viewpoint alone); "
                      "both  = frame advances AND camera orbits together, one "
                      "full turn across the 150 frames")
_ap.add_argument('--n-frames', type=int, default=150)
_ap.add_argument('--pin-frame', type=int, default=75, help='sweep=angle: which frame')
_ap.add_argument('--n-angles', type=int, default=120)
_ap.add_argument('--elev', type=float, default=15.0)
_ap.add_argument('--radius', type=float, default=2.0)
_ap.add_argument('--fps', type=int, default=15)
_ap.add_argument('--no-gt', action='store_true',
                 help='sweep=angle: drop the GT panel. The GT is a single-camera '
                      '2D video and cannot orbit, so in a turntable it is a '
                      'static distraction.')
_ap.add_argument('--lora-scale', type=float, default=1.0,
                 help='apply the TRAINED adapter at strength s. delta = s*B*(A x), '
                      'so s=0 is exactly frozen and s=1 is the run as trained. '
                      'Scales B only; A is untouched, so the per-voxel mixing is '
                      'unchanged and only the magnitude moves. DRaFT (ICLR 2024) '
                      'reports this as its most effective control on '
                      'over-optimisation.')
_ap.add_argument('--turns', type=float, default=1.0,
                 help='sweep=both: how many full revolutions across the whole '
                      'sequence. 2 gives 4.8 deg/frame instead of 2.4, so every '
                      'angle is seen at two different points in time.')
_ap.add_argument('--out-dir', default=None)
ARGS = _ap.parse_args()

RUN_DIR = (_HERE / 'runs' / ARGS.run).resolve()
assert RUN_DIR.exists(), f'no such run: {RUN_DIR}'
CFG = json.load(open(RUN_DIR / 'config.json'))

# The training script parses argv at import time, so hand it the config the run
# was actually trained with. Anything else would build a different registry than
# the checkpoint expects.
sys.argv = [
    'rung14_v1_crossattn_lora.py',
    '--rank', str(CFG['rank']),
    '--seed', str(CFG['seed']),
    '--targets', 'qkvo' if len(CFG['targets']) == 3 else 'qkv',
    '--epochs', str(CFG['epochs']),
]
import rung14_v1_crossattn_lora as R          # noqa: E402

import numpy as np                             # noqa: E402
import torch                                   # noqa: E402
from PIL import Image, ImageDraw               # noqa: E402

DEVICE = R.DEVICE
_SUB = {'time': 'temporal', 'angle': 'orbit', 'both': 'orbit_both'}[ARGS.sweep]
OUT = Path(ARGS.out_dir) if ARGS.out_dir else (RUN_DIR / _SUB)
(OUT / 'frames').mkdir(parents=True, exist_ok=True)


def orbit_extrinsics(yaw_deg, elev_deg, radius):
    """
    Camera on a sphere, looking at the origin, in the convention v1's
    MeshRenderer expects. GATE-cam below asserts that yaw=0, elev=0 reproduces
    the confirmed front-view EXTRINSICS, so a 'the texture is a sticker' verdict
    can never be an artefact of a wrong camera.
    """
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = right, -up, fwd
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32)


def main():
    print('=' * 88)
    print(f'RUNG14-v1 RENDER  sweep={ARGS.sweep}  run={ARGS.run}')
    print('=' * 88, flush=True)

    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained(R.PRETRAINED)
    pipeline.to(DEVICE)
    flow_model = pipeline.models['slat_flow_model']
    dec_model = pipeline.models['slat_decoder_mesh']
    for p in flow_model.parameters(): p.requires_grad_(False)
    for p in dec_model.parameters():  p.requires_grad_(False)

    ref = Image.open(R.GT_FRAMES_DIR / 'frame_0075.png').convert('RGB')
    cond_struct = pipeline.get_cond([ref])
    torch.manual_seed(R.STRUCT_SEED)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1)
    assert coords.shape[0] == 7301, f'N_vox={coords.shape[0]} != 7301'
    del cond_struct; gc.collect(); torch.cuda.empty_cache()

    dino = pipeline.models['image_cond_model']
    registry = R.XAttnLoRARegistry(flow_model, R.ACTIVE_BLOCKS, CFG['rank'],
                                   tuple(CFG['targets'])).to(DEVICE)
    st = torch.load(RUN_DIR / 'lora_ckpts' / ARGS.ckpt, map_location=DEVICE,
                    weights_only=False)
    # rung14_v1_crossattn_lora.py saves the adapter under 'registry_state'
    # alongside epoch/run_id/best_psnr/loss_scale/optimizer. Guessing the key and
    # silently falling back to the whole dict is how this failed the first time,
    # so the key is required and the tensor count is checked: 24 blocks x 3
    # targets x (A, B) = 144.
    assert 'registry_state' in st, f'no registry_state in ckpt; keys={list(st)}'
    sd = st['registry_state']
    assert len(sd) == len(R.ACTIVE_BLOCKS) * len(CFG['targets']) * 2, (
        f'{len(sd)} tensors in ckpt, expected '
        f'{len(R.ACTIVE_BLOCKS)*len(CFG["targets"])*2}')
    registry.load_state_dict(sd)
    registry.eval()
    if ARGS.lora_scale != 1.0:
        with torch.no_grad():
            for _blk in registry.blocks.values():
                for _n in _blk.targets:
                    getattr(_blk, f'lora_{_n}').B.mul_(ARGS.lora_scale)
        print(f'[LORA-SCALE] B scaled by {ARGS.lora_scale}  '
              f'(0 = frozen, 1 = as trained)', flush=True)
    n_par = sum(p.numel() for p in registry.parameters())
    print(f'[LORA] {ARGS.ckpt}  epoch={st.get("epoch", "?")}  '
          f'best_psnr={st.get("best_psnr", float("nan")):.4f}  {n_par:,} params '
          f'blocks={len(R.ACTIVE_BLOCKS)} targets={tuple(CFG["targets"])}', flush=True)

    torch.manual_seed(CFG['seed'])
    noise = torch.randn(coords.shape[0], flow_model.in_channels, device=DEVICE)

    renderer = R.make_renderer()

    # ── GATE-cam ────────────────────────────────────────────────────────────
    e0 = orbit_extrinsics(0.0, 0.0, 2.0)
    d = float((e0 - R.EXTRINSICS).abs().max())
    print(f'\n[GATE-cam] |orbit(0,0,2) - confirmed EXTRINSICS| = {d:.3e}')
    assert d < 1e-5, (
        f'GATE-cam FAILED ({d:.3e}): the orbit camera does not reproduce the '
        f'confirmed front view at yaw=0, so any judgement about the texture '
        f'holding up under rotation would be an artefact of the camera.')
    print('[GATE-cam] PASSED', flush=True)

    cond_pin = None
    if ARGS.sweep == 'time':
        items = [(f, None) for f in range(1, ARGS.n_frames + 1)]
        label = 'frame'
    elif ARGS.sweep == 'angle':
        items = [(ARGS.pin_frame, i * 360.0 / ARGS.n_angles)
                 for i in range(ARGS.n_angles)]
        label = 'yaw'
        # encode_frame() returns [N_tok, C] (it squeezes the batch); the flow's
        # cross-attention needs [1, N_tok, C], which training adds at the call
        # site (rung14_v1_crossattn_lora.py:769, :811). Without it the kv reshape
        # reads N_tok as the batch and fails:
        #   shape '[1374, 2048, 2, 16, -1]' invalid for input of size 2813952
        cond_pin = R.encode_frame(dino, ARGS.pin_frame).unsqueeze(0)
    else:  # both — ARGS.turns full revolutions spread across the whole sequence
        n = ARGS.n_frames
        items = [(f, (f - 1) * 360.0 * ARGS.turns / n) for f in range(1, n + 1)]
        label = 'frame+yaw'

    verts_fz, verts_lo, mismatches = [], [], 0
    for k, (fr, yaw) in enumerate(items, 1):
        cond = (cond_pin if cond_pin is not None
                else R.encode_frame(dino, int(fr)).unsqueeze(0))
        ext = (None if yaw is None
               else orbit_extrinsics(yaw, ARGS.elev, ARGS.radius).to(DEVICE))

        slat_fz = R.full_denoise_nograd(flow_model, noise, coords, cond, None)
        slat_lo = R.full_denoise_nograd(flow_model, noise, coords, cond, registry)
        with torch.no_grad():
            mesh_fz = dec_model(slat_fz)[0]
            mesh_lo = dec_model(slat_lo)[0]
            # render_mesh() reads the module-level EXTRINSICS rather than taking
            # one, so the orbit camera is installed by swapping that global and
            # restoring it. Going through render_mesh (instead of calling the
            # renderer directly) keeps the alignment and the degenerate-face
            # filter identical to training — the rung13 lesson was that a
            # diagnostic which bypasses the production path certifies nothing.
            _saved_ext = R.EXTRINSICS
            if ext is not None:
                R.EXTRINSICS = ext
            try:
                col_fz, _ = R.render_mesh(mesh_fz, renderer)
                col_lo, _ = R.render_mesh(mesh_lo, renderer)
            finally:
                R.EXTRINSICS = _saved_ext
        verts_fz.append(int(mesh_fz.vertices.shape[0]))
        verts_lo.append(int(mesh_lo.vertices.shape[0]))
        if verts_fz[-1] != verts_lo[-1]:
            mismatches += 1

        panels = []
        # The GT is a single fixed-camera 2D video and cannot orbit. Beside a
        # rotating render it still advances in TIME with the frame index, but
        # stays at yaw 0 — so it is a control for the texture's evolution, NOT
        # for the viewpoint. The panel label states that on every frame so the
        # video cannot be misread. --no-gt drops it.
        if not ARGS.no_gt:
            gt, _ = R.load_gt(int(fr))
            panels.append((gt, 'GT video' if yaw is None
                               else 'GT video (fixed cam, yaw 0)'))
        panels.append((col_fz, f'frozen  ({verts_fz[-1]:,} v)'))
        _sl = '' if ARGS.lora_scale == 1.0 else f' s={ARGS.lora_scale:g}'
        panels.append((col_lo, f'x-attn LoRA{_sl}  ({verts_lo[-1]:,} v)'))
        R.make_strip(panels).save(OUT / 'frames' / f'{k:04d}.png')

        del slat_fz, slat_lo, mesh_fz, mesh_lo, col_fz, col_lo
        if k % 10 == 0 or k == 1:
            _w = f'f{fr:04d}' + ('' if yaw is None else f' yaw={yaw:5.1f}')
            print(f'  {k:3d}/{len(items)}  {_w}  '
                  f'verts fz={verts_fz[-1]:,} lora={verts_lo[-1]:,}', flush=True)
        torch.cuda.empty_cache()

    print(f'\n[GEOMETRY] vertex-count mismatches frozen vs lora: {mismatches}/{len(items)}')
    print(f'           frozen  min={min(verts_fz):,} max={max(verts_fz):,}')
    print(f'           lora    min={min(verts_lo):,} max={max(verts_lo):,}')
    json.dump(dict(sweep=ARGS.sweep, n=len(items), ckpt=ARGS.ckpt,
                   epoch=st.get('epoch'), verts_frozen=verts_fz,
                   verts_lora=verts_lo, mismatches=mismatches,
                   pin_frame=ARGS.pin_frame if ARGS.sweep == 'angle' else None),
              open(OUT / 'render.json', 'w'), indent=2)

    name = f'RUNG14V1_{ARGS.sweep.upper()}_frozen_vs_xattn.mp4'
    vid = OUT / name
    pr = subprocess.run(['/usr/bin/ffmpeg', '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(ARGS.fps),
                    '-i', str(OUT / 'frames' / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)],
                   check=True)
    print(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
