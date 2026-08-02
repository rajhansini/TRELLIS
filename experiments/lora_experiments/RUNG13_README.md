# Rung 13 — Registering the mesh to the video before adapting appearance

**Read this first if you are new to the project.** It covers what we are building, how
TRELLIS works, what went wrong, how we found it, what we changed, and what is still
broken. Every number here was measured; nothing is estimated.

---

## 1. What we are building

Given **one video of one object**, produce **one 3D mesh whose texture evolves over
time** — the geometry stays fixed, the appearance changes frame to frame.

> One mesh. A series of textures. **Not** a series of meshes.

That distinction matters and it drives every design decision below. We are *not*
trying to make the geometry move. A mesh that changes shape every frame would be a
different (harder, and for us wrong) problem. We want a single stable asset whose
surface appearance is dynamic — lava cooling and reheating on a teapot, in our case.

Our test data: 150 frames of a video-generated "lava teapot", one fixed camera.

Constraint: **the TRELLIS checkpoint is never fine-tuned.** Everything we add is a
small adapter (LoRA) on a frozen backbone.

---

## 2. TRELLIS in five minutes

TRELLIS turns a single image into a 3D asset. Six stages:

```
INPUT: one RGB image
   │
   ▼
[A] CONDITIONING
    rembg → crop → resize 518²  →  DINOv2 ViT-L/14  →  patch tokens (1374 × 1024)
    This is the only place the image enters the system.
   │
   ▼
[B] SPARSE STRUCTURE FLOW              model: sparse_structure_flow_model
    noise (8 × 16³) ← torch.manual_seed(seed)
    rectified-flow ODE, 12 steps, cross-attends to the DINOv2 tokens
    → occupancy grid (64³) → threshold
    → coords (N × 4)  =  which voxels exist.   N = 7,301 for our teapot
   │
   ▼
[C] SLAT FLOW                          model: slat_flow_model   (24 blocks)
    noise (N × 8) ← same seed stream
    rectified-flow ODE, 25 steps, sparse transformer over the voxels
    each block:  self-attention over voxels   ← 3D coherence
                 cross-attention to image tokens  ← image evidence enters
    → slat (N × 8):  eight numbers per occupied voxel.  This is "the latent".
   │
   ▼
[D] MESH DECODER                       model: slat_decoder_mesh  (12 blocks)
    input_layer            8 → 768
    12 × windowed self-attention blocks, 7,301 voxels throughout
    upsample ×2            7,301 → 58,408 → 467,264 voxels
    out_layer              SparseLinear 96 → 101
   │
   ▼
[E] FLEXICUBES  (res 256)  →  vertices, faces, per-vertex colours
   │
   ▼
[F] RENDER  nvdiffrast, antialiased
```

**Two facts you will need repeatedly:**

- There is **no cross-attention in the decoder**. The image only influences stages B
  and C. By stage D the 3D reasoning is already done.
- There are **no UV maps and no texture images**. Colour is a **per-vertex attribute**
  interpolated across triangles (`dr.interpolate(mesh.vertex_attrs[:, :3], ...)` in
  `trellis/renderers/mesh_renderer.py`). "The texture is misaligned" can therefore
  never mean a UV bug — there are no UVs to get wrong.

### The `out_layer` channel layout

`out_layer` emits 101 numbers per fine voxel. From
`trellis/representations/mesh/cube2mesh.py`:

| channels | name | meaning |
|---|---|---|
| `[0:8]` | **sdf** | inside/outside value at the cube's 8 corners |
| `[8:32]` | **deform** | xyz displacement of each corner |
| `[32:53]` | **weights** | FlexiCubes α/β/γ — controls topology |
| `[53:101]` | **colour** | 8 corners × 6 = 24 albedo + 24 shading normal |

**Vertex positions are computed from `sdf` and `deform` only.** That is the hook
that makes "freeze geometry, adapt appearance" provable rather than hopeful.

---

## 3. How we adapt appearance without touching geometry

We put a LoRA on the **12 decoder blocks** (all four sublayers each: `attn.to_qkv`,
`attn.to_out`, `mlp.mlp[0]`, `mlp.mlp[2]`), rank 4 → **589,824 trainable parameters**.
The TRELLIS weights are frozen.

But a block-level LoRA changes *all* 101 out_layer channels, including `sdf` — it
would move the mesh. So we use a **two-pass splice**
(`rung5_colonly_lora.py::colonly_forward`, inherited unchanged by rung13):

```python
# PASS 1 — frozen decoder, no LoRA
with torch.no_grad():
    h1 = dec_model.out_layer.register_forward_hook(_hook('frozen'))
    dec_model(slat_norm)
    h1.remove()

# PASS 2 — same decoder, LoRA hooks active on all 12 blocks
h2 = dec_model.out_layer.register_forward_hook(_hook('lora'))
with dec_block_lora_ctx(dec_model, registry):
    dec_model(slat_norm)
h2.remove()

# SPLICE — geometry from pass 1, colour from pass 2
frozen_geom = captured['frozen'].feats[:, :53].detach()          # sdf+deform+weights
lora_col    = captured['lora'  ].feats[:, 53:101].clamp(-9, 8)   # colour, HAS GRAD
mixed       = torch.cat([frozen_geom, lora_col], dim=1)
mesh        = dec_model.to_representation(frozen_h.replace(mixed))
```

Two full decoder passes per training step. Expensive, but it buys a guarantee:
**gradients reach only the colour channels, so the mesh cannot move.**

Verified, not assumed — across a full 360° orbit and all 150 frames:

```
[GEOMETRY] vertex-count mismatches: 0/150
[AREA]     max |frozen − ours| over all angles = 0 px
```

**Why upstream (decoder blocks) instead of at `out_layer` itself?** `out_layer` is a
pointwise map — voxel *v* out depends only on voxel *v* in. It can recolour a point
but cannot *move* a pattern. Adapters placed there reconstruct the video at ~10.6 dB;
block-level adapters, which sit before windowed attention and two conv upsamplers and
so can redistribute information spatially, reach ~21.3 dB. Same rank, same seed, only
the hook moved.

---

## 4. The bug

For weeks the result looked good from the training camera and fell apart under
rotation — the lava behaved like a **sticker** applied to the front of the object.
Several hypotheses were tested and killed:

| hypothesis | test | verdict |
|---|---|---|
| UV mapping is wrong | read the renderer | **dead** — no UVs exist |
| the adapter learned `colour ≈ f(height)` | R² of Δ vs the vertical axis | **dead** — 0.010–0.032, no higher than the frozen colour's own 0.004–0.029 |
| it tracks depth from the camera | R² vs the camera axis | **dead** — 0.006–0.011 |
| unsupervised voxels get a wilder edit | \|Δ\| on seen vs unseen vertices | **dead** — ratio 1.02, i.e. *identical* |

What did show up: **R² = 0.16–0.23 against the training camera's image left–right
axis**, versus ~0.00 for the frozen colour. The edit was organised by the *view*, not
by the surface.

That pointed at registration, and measuring it directly confirmed it. Over 15 frames
in the training view:

| | centroid x | centroid y | width | height | area |
|---|---|---|---|---|---|
| GT teapot (video) | 265.1 | 270.8 | 319 | 179 | 30,961 |
| rendered mesh | 248.9 | 271.5 | **355** | 181 | 34,483 |

**The render was 11.2% wider, the same height, and 16 px to the side.** Silhouette
IoU **0.76**. A crude 2D scale+shift lifted that to 0.909, showing ~62% of the
mismatch was *pose*, not genuine shape difference.

### Why that makes the texture slip

The loss compares render and video **pixel by pixel at the same screen position**.

Take pixel (300, 250):

- in the **video** it shows the teapot's **spout**
- in the **render**, because the mesh sits 16 px over, it shows the teapot's **body**

The loss says *"make pixel (300,250) match"*. The adapter can only change colour,
never position. So it paints **spout colour onto the body** — and does so everywhere,
consistently off by the same 16 px.

From the training camera this is invisible: the wrong part is standing exactly where
the right part should be, so the image matches. Rotate the camera, the body moves
away, and you see spout-coloured paint sitting on the body.

**The paint was never on the right part of the object. It only looked that way from
one angle.**

---

## 5. The fix

Solve a **3D similarity transform** that registers the mesh to the video, and apply it
to the mesh before rendering:

```
v' = s · R(rotvec) · (v − centre) + centre + t          7 degrees of freedom
```

Solved values (`visibility/alignment/alignment.json`):

```
s      = 0.8903
rotvec = [-0.1091, 0.0015, -0.0530]     |rotvec| = 0.1213 rad = 7.0°
t      = [ 0.0412, -0.0472,  0.0013]
```

### Two design decisions worth understanding

**(a) The objective is silhouette overlap only — it never looks at colour.**

```python
def soft_iou_loss(soft_mask, gt_mask):
    g = gt_mask.float()
    inter = (soft_mask * g).sum()
    union = soft_mask.sum() + g.sum() - inter
    return 1.0 - inter / (union + 1e-8)
```

*IoU = intersection over union*: overlap the two outlines, divide the area where both
agree by the area either covers. 1.0 = identical, 0 = no overlap.

Colour is the artifact we are trying to explain, so if the fix were fitted using
colour the argument would be circular. Fitting on outlines alone avoids that.

**(b) We move the object, not the camera.** A camera tweak or a 2D image warp fixes
the one view and breaks under rotation — it corrects the projection, not the thing
being projected. A 3D object transform is correct from every viewpoint, which is the
entire point.

### How the solve works — `visibility/solve_mesh_alignment.py`

Two stages, because silhouette gradients only exist at the object's boundary:

1. **Coarse grid** over (scale, tx, ty) — 585 combinations, hard IoU, no gradients.
   Needed because Adam started from identity sits in a flat region where the two
   outlines barely touch and there is nothing to descend.
2. **Adam**, lr 3e-3, 300 iterations, all 7 DOF. `nvdiffrast` antialiases its mask
   (`dr.antialias`), so the rendered silhouette is differentiable with respect to
   vertex positions and gradients flow back to the 7 numbers. Converged by iteration
   50.

Rotation is parameterised as a rotation vector via Rodrigues' formula and taken
**about the mesh centroid**, so that scale, rotation and translation stay close to
independent — rotating about the origin would smear rotation into translation.

*Why not ICP or PnP?* ICP needs 3D↔3D correspondences and we only have a 2D video.
PnP needs 2D↔3D keypoint matches and a smooth untextured teapot has none. Silhouettes
are the only correspondence-free signal available.

**Validation** — fitted on 5 frames, scored on 15 it never saw:

```
silhouette IoU  0.7596 → 0.9051     (+0.1455, 60.5% of the gap to 1.0)
residual        0.0949   ← genuine shape difference; no pose fixes this
```

---

## 6. Confirming the mechanism before trusting it

A plausible story is not evidence. Before spending a training run we made a
**falsification test with thresholds fixed in advance**
(`visibility/test_alignment_compensation.py`).

**The prediction.** If the adapter really contains a compensating shift, then
rendering the *already-trained* adapter on the *newly aligned* mesh double-counts the
compensation and should make things **worse in a specific direction**:

```
today   mesh −16 px  +  texture +24 px   →  cancels      →  NCC peak (0,  0)
test    mesh aligned +  texture +24 px   →  overshoots   →  NCC peak (0, +24)
```

**Thresholds set before running:** ≥ 8 px = confirmed, ≤ 3 px = refuted, in between =
inconclusive, do not build on it.

**Result** (2D normalised cross-correlation against the GT):

| frame | dx on unaligned mesh | dx on aligned mesh | moved |
|---|---|---|---|
| 75 | 0 | +19 | +19 |
| 120 | 0 | +5 | +5 |
| 150 | 0 | +8 | +8 |

**Mean +10.7 px → CONFIRMED.** The texture had a compensating displacement baked into
it, exactly as required.

---

## 7. What changed in the code

rung13 is a **copy** of `rung5_colonly_lora.py` with one change. The parent file is
untouched and still runs.

### The change — apply the transform inside `render_mesh`

`render_mesh` is the single choke point every render goes through, so training,
evaluation, diagnostics and gates all see identical geometry.

```python
def align_vertices(v):
    """v' = s*R(v - centre) + centre + t.  Fixed transform, never learned."""
    return ALIGN_S * ((v - _AC) @ _AR.T) + _AC + _AT


def render_mesh(mesh, renderer, aligned=True):
    ext, intr = EXTRINSICS.to(DEVICE), INTRINSICS.to(DEVICE)
    mesh = filter_degenerate_faces(mesh)
    _saved = mesh.vertices
    if aligned:
        mesh.vertices = align_vertices(mesh.vertices)
    res = renderer.render(mesh, ext, intr, return_types=['color', 'mask'])
    mesh.vertices = _saved                      # restore, never mutate the caller's mesh
    mask = res['mask'].unsqueeze(0)
    return res['color'] * mask + (1.0 - mask), mask
```

The transform is loaded from `alignment.json` and is **fixed — never learned**.

### A necessary consequence, not a second variable

`debug_results/step8_mesh/mask.png` is a **cached silhouette of the unaligned render**.
Once the mesh moves, that file is stale and the loss region it defines is wrong. So
the static mask is recomputed from the aligned frozen mesh:

```
stale mask.png : 34,775 px
aligned mask   : 28,554 px
IoU(stale, aligned) = 0.7257     ← low is EXPECTED; that IS the misregistration
```

This is why **PSNR is not directly comparable to the parent's** — it is measured over
a different region. SSIM and LPIPS are computed over the full image and *are*
comparable.

### Gates — the run refuses to start if any fails

| gate | asserts | why |
|---|---|---|
| `GATE-align` | silhouette IoU improves by > 0.05 | if the transform is not reaching the renderer, rung13 is rung5 with a new name and every number would be meaningless |
| `GATE-plain` | at `B = 0`, our forward == frozen decoder | the adapter starts as the identity |
| `GATE 2` | gradients reach all 48 LoRA `B` matrices | catches a silently dead adapter |
| `GATE 0` | parameter count == `rank × 12288 × 12` | catches a mis-built registry |

Observed at startup:

```
[tight GT mask] 30,936 px = 11.5% of the image  (the leaky one would be ~99.7%)
[GATE-align]    unaligned = 0.7611   aligned = 0.9054   (+0.1443)   PASSED
[GATE-plain]    PASSED   (both via render_mesh — the training path)
[GATE 2]        PASSED
```

The in-script 0.7611 → 0.9054 matches the standalone solver's 0.7596 → 0.9051 on
held-out frames. Two independent code paths, same answer.

> **Lesson worth internalising:** three separate bugs during this work were caught by
> gates, and all three were bugs in the *diagnostics*, not the method. A gate that
> prints `PASSED` while testing the wrong code path is worse than no gate. Always
> check that a gate exercises the path production actually uses.

---

## 8. Results

Retraining with the registered mesh, **nothing else changed**:

| | PSNR | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| parent `rung5_colonly` | 21.302 | 0.9471 | 0.0345 |
| **rung13 aligned** | **22.645** | **0.9556** | **0.0208** |

**40% lower perceptual error.** SSIM and LPIPS are full-image and directly
comparable; PSNR carries the region caveat above.

Geometry provably unchanged: **0/150** frames with a vertex-count difference, area
ratio **1.0000**.

Colour actually learned, object pixels, frame 75:

```
frozen  [ 80.1, 29.7,  8.2]   ← starting point
ours    [106.4, 46.7, 25.3]
GT      [108.3, 47.6, 25.1]   ← target
```

Sanity check against the obvious failure mode (is the output secretly a copy of the
GT?): only **378 of 268,324** pixels exactly equal the GT, while **239,766** exactly
equal the frozen render — the background, untouched by a colour-only edit. The
rendered silhouette is 28,383 px versus the GT's 31,037: it has TRELLIS's outline, not
the video's.

---

## 9. What this does **not** fix

Be honest about this in any writeup.

- **Coverage.** One camera supervises only **16.8%** of the mesh's vertices, and the
  adapter edits 100% of them at the same strength (measured ratio 1.02). The other
  83% receives an edit the loss never evaluated. Turntables still degrade away from
  the training view.
- **Residual shape difference.** After alignment, 0.0949 IoU of mismatch remains.
  TRELLIS genuinely reconstructed a slightly different teapot. No pose fixes that.
- **Generalisation is untested.** Held-out frames are every 10th frame — temporal
  interpolation between frames the model saw. Held and all-frame PSNR are identical
  (22.646 vs 22.645), so there is no overfitting gap, but this is not a test of novel
  views or novel sequences.
- **One object, one video, no baseline.** Everything here is a single scene.

The claim the results support is *"faithful dynamic texture from the captured view,
with geometry provably unchanged."* Not *"solved."*

---

## 10. Reproducing it

```bash
cd /net/projects/ranalab/rajhansini/TRELLIS
export SPCONV_ALGO=native ATTN_BACKEND=xformers

# 1. solve the alignment (writes visibility/alignment/alignment.json)
sbatch experiments/lora_experiments/visibility/jobs/align.sbatch

# 2. confirm the compensation mechanism — preregistered, can refute the theory
sbatch experiments/lora_experiments/visibility/jobs/comp_test.sbatch

# 3. train (30 epochs, ~70 min on an A40; resumes from its per-epoch checkpoint)
sbatch experiments/lora_experiments/visibility/jobs/rung13.sbatch

# 4. the dynamic/temporal video — fixed camera, 150 frames
sbatch experiments/lora_experiments/visibility/jobs/rung13_temporal.sbatch

# 5. the turntable — frame pinned, camera orbits 360°
sbatch experiments/lora_experiments/visibility/jobs/colonly_orbit.sbatch
```

Step 3 will refuse to start if step 1 has not run.

**Practical notes for whoever runs this next:**

- Every job is a real `.sbatch` file, never `sbatch --wrap`. `--wrap` runs under
  `/bin/sh`, which is `dash` on this cluster, and `dash` has no `set -o pipefail` —
  two jobs died in 0 seconds before we worked that out.
- Walltime is capped at **4 hours** on every partition. rung13 fits in ~70 min.
- `--requeue` plus per-epoch checkpoints means a preempted job resumes rather than
  restarting.
- The SLaT cache is copied from the parent run and verified bit-identical, so both
  runs consume the same latents and the alignment is genuinely the only difference.

---

## 11. File map

| file | what it is |
|---|---|
| `rung13_aligned_colonly_lora.py` | the run. One-variable diff from its parent |
| `rung5_colonly_lora.py` | the parent, unmodified — the baseline rung13 is compared against |
| `visibility/solve_mesh_alignment.py` | the 7-DOF solver (coarse grid + Adam through nvdiffrast) |
| `visibility/alignment/alignment.json` | the solved transform + per-frame IoU before/after |
| `visibility/test_alignment_compensation.py` | the preregistered falsification test |
| `visibility/render_colonly_comparison.py` | 150-frame fixed-camera video (the dynamic effect) |
| `visibility/render_colonly_orbit.py` | 360° turntable (the 3D-consistency check) |
| `visibility/jobs/*.sbatch` | launch scripts for each of the above |

Upstream files worth reading when you need ground truth about the model:

| file | why |
|---|---|
| `trellis/representations/mesh/cube2mesh.py` | the `out_layer` channel layout |
| `trellis/representations/mesh/utils_cube.py` | how per-cube values become vertices; `con_loss` |
| `trellis/models/structured_latent_vae/decoder_mesh.py` | decoder forward — blocks, then upsamplers, then `out_layer` |
| `trellis/modules/sparse/transformer/modulated.py` | the SLaT flow block: `self_attn` + `cross_attn` |
| `trellis/renderers/mesh_renderer.py` | proves colour is per-vertex and there are no UVs |

---

## 12. Where this is going next

The open problem is **coverage**: one camera, 16.8% of the surface supervised, and no
mechanism to reach the rest.

The promising direction — following **Fuse3D** (SIGGRAPH Asia 2025), which controls
TRELLIS by scaling **cross-attention logits** in the SLaT flow, entirely training-free
— is that *TRELLIS's own cross-attention map already encodes a 2D↔3D correspondence*.
Voxels the camera never saw still attend to image tokens. If unseen voxels attend to
the same tokens as seen ones, the appearance edit can be propagated along that
correspondence instead of hoping the decoder generalises.

Note the contrast with what we do here: our adapter sits in the **decoder**, *after*
all 3D reasoning, which is structurally why it cannot propagate. An intervention in
the **SLaT flow's cross-attention** sits *before* 24 blocks of self-attention and 25
ODE steps, and would inherit TRELLIS's own 3D propagation.
