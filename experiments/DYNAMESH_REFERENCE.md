# DynaMesh — complete reference

Everything about the geometry-vs-texture problem, end to end. Every loss written
out, every number pulled from the run directories on disk, every mechanism traced
to the file and line it lives in.

Last updated 2026-07-30, after rung9.

---

# PART 1 — THE GOAL

Make a TRELLIS-generated 3D mesh's **texture and geometry evolve over time**,
supervised by a single video, without retraining the backbone.

Input: 150 frames of a Kling-generated video of a lava teapot, one fixed camera.
Output: 150 meshes that share a topology and differ smoothly over time.

Constraint: the TRELLIS checkpoint stays bit-identical. Everything we add is a
small adapter.

---

# PART 2 — THE PIPELINE, STOCK

```
INPUT: one RGB image
   │
   ▼
[A] CONDITIONING
   rembg → crop → resize 518²
   DINOv2 ViT-L/14 → patch tokens (1 × 1374 × 1024)
   │
   ▼
[B] SPARSE STRUCTURE FLOW              model: sparse_structure_flow_model
   noise (1 × 8 × 16³) ← torch.manual_seed(seed), FIRST draw
   rectified-flow ODE, 12 steps, cfg 7.5, cross-attends to tokens
   → latent (1 × 8 × 16³)
   → sparse_structure_decoder (3D conv) → occupancy (1 × 1 × 64³)
   → threshold > 0
   → coords (N × 4) = [batch, x, y, z]        N = 7,301
   OUTPUT: WHICH voxels exist. Discrete.
   │
   ▼
[C] SLAT FLOW                          model: slat_flow_model
   noise (N × 8) ← same seed stream, SECOND draw
   rectified-flow ODE, 25 steps, cfg 3.0, sparse transformer on coords
   → slat (N × 8)
   OUTPUT: 8 numbers per occupied voxel. Continuous. The latent.
   │
   ▼
[D] MESH DECODER                       model: slat_decoder_mesh
   input_layer            8 → 768
   self.blocks   12 × SparseResBlock3d, WINDOWED SELF-ATTENTION w=8³
                 shifted by 4 on odd blocks; res 64; 7,301 voxels throughout
   upsample[0]   SparseSubdivideBlock3d  768→192   res  64→128    7,301 → 58,408
   upsample[1]   SparseSubdivideBlock3d  192→96    res 128→256   58,408 → 467,264
   out_layer     SparseLinear  96 → 101
   │
   ▼
[E] FLEXICUBES  (res 256)
   → vertices, faces, vertex colours
   │
   ▼
[F] RENDER  nvdiffrast, dr.antialias on colour AND mask
```

Verified in `trellis/models/structured_latent_vae/decoder_mesh.py:162-168` — all
12 attention blocks run **before** both upsamplers.

**There is no cross-attention in the decoder.** The image conditioning enters at
stages B and C only. The decoder sees the SLaT and nothing else.

---

# PART 3 — OUR PIPELINE vs STOCK

| Stage | Stock | Ours | Why |
|---|---|---|---|
| [A] | once, one image | **150×**, one per frame | the video enters here and only here |
| [B] | once per image | **once, frame 75 only**; `coords` cached and reused for all 150 | free coords jump discontinuously frame to frame (13.3× the pinned flicker, measured) |
| [B] seed | 42 | `STRUCT_SEED = 42` | same |
| [C] | once, fresh noise | **150×**, but with a **fixed** noise vector (N×8, seed 6) reused every frame | holds the random draw constant so the only per-frame variation is the conditioning |
| [C] out | `slat` (N×8) | `slats` (150 × 7,301 × 8) cached to `slat_cache.npz` | A/B/C never run during training |
| [D] backbone | inference | frozen, **but under autograd** | activations retained so gradient reaches the hook |
| [D] `out_layer` | plain Linear | **+ LoRA**: `raw + B·A·x_v` | the only modified op in the network |
| [E] | same | same | untouched |
| [F] | turntable for output | **one fixed camera**, 512² | training is single-view |
| loss | **none** — TRELLIS is feed-forward | MSE + LPIPS, intersection-masked | this stage does not exist upstream. It is the entire contribution |

---

# PART 4 — THE `out_layer` CHANNELS

`SparseLinear(96 → 101)`. From `trellis/representations/mesh/cube2mesh.py`
`_calc_layout()`:

| channels | name | shape | what it is |
|---|---|---|---|
| `[0:8]` | **sdf** | 8 × 1 | this cube's opinion about its 8 corners |
| `[8:32]` | **deform** | 8 × 3 | xyz displacement of each of the 8 corners |
| `[32:53]` | **weights** | 21 | FlexiCubes: β(12) + α(8) + γ(1) |
| `[53:101]` | **colour** | 8 × 6 | 24 albedo + 24 shading normal |

**Vertex positions read only `sdf` and `deform`.** `[32:53]` steers *which*
triangles exist. `[53:101]` is pure appearance.

`out_layer` is the last operation in the network that does **not mix voxels**.
Voxel *v*'s output depends only on voxel *v*'s input. That is the whole reason
the hook lives there.

---

# PART 5 — HOW GEOMETRY IS ACTUALLY COMPUTED

This is the part that broke rung9. Read it slowly.

### 5.1 The grid

Space is a 256×256×256 grid of **cubes**. Each cube has 8 corners. Each corner
gets one sdf number:

- negative = inside the teapot
- positive = outside

The surface is drawn where the sign flips. Vertex position along an edge is a
**linear interpolation** between the two corner values — which is why sdf is
differentiable w.r.t. vertex position.

### 5.2 Cubes share corners

A corner is a point in space, not a property of one cube. In 3D, **8 cubes touch
every interior corner.**

But `out_layer` emits **8 numbers per cube**, not one per corner. So 8 cubes each
write their own number for the same point.

### 5.3 TRELLIS averages them

`trellis/representations/mesh/utils_cube.py:26-44`, `cubes_to_verts`:

```python
reduced = zeros(num_verts, M).scatter_add(0, idx, src)
cnt     = bincount(idx).clamp(min=1)
reduced = reduced / cnt          #  reduce='mean'
```

The averaged value is what draws the surface.

### 5.4 The term that makes the average faithful

`utils_cube.py:47-54`:

```python
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
```

`con_loss` = *how far each cube's own opinion is from the consensus*. TRELLIS
trained with it, so neighbouring cubes learned to agree, so the average is
faithful to what every cube wanted, so the surface is smooth.

**It is behind `if training:`.** So is `L_dev`. So is the weight penalty
(`cube2mesh.py:158-163`). `training` comes from `self.training` on the decoder
module, and we call `dec_model.eval()`.

**In every run so far, all three were 0.0 — not down-weighted, never computed.**

### 5.5 Deform

`utils_cube.py:73-74`:

```python
x_nx3 = v_pos/res - 0.5 + (1 - 1e-8)/(res*2) * tanh(deform)
```

`tanh` bounds it: a corner can move at most half a cube width. So `deform` can
never tear the grid; only `sdf` can move the surface across cube boundaries.

Also `sdf += sdf_bias` where `sdf_bias = -1.0/res`.

### 5.6 `training=True` changes the MESH, not just the losses

`flexicubes.py:365-389`. In eval, each quad splits into **2 triangles** along
whichever diagonal `gamma` prefers — a hard comparison. In training, a **centre
vertex is inserted** and the quad becomes **4 triangles** — differentiable.

Consequence: you cannot flip the flag globally and keep numbers comparable.
Train with `training=True`, measure with `training=False`.

---

# PART 6 — EVERY LOSS, WRITTEN OUT

## 6.1 The masks

```
gt_mask          = min over RGB of gt  <  0.95
live_render_mask = renderer mask > 0.5          (per frame, from nvdiffrast)
static_rmask     = mask.png > 128               (one frame, reused; metrics only)
```

**Why 0.95 and not 0.99.** The Kling background is off-white — measured corner
minimum per channel `[0.9843, 0.9882, 0.9765]`. v4 used `(gt < 0.99).any(dim=0)`,
which fires on the background and returns **99.7% of the image**. Intersecting
with that would have dropped 437 px of 34,416 and made the intersection loss a
no-op. `min(RGB) < 0.95` gives 11.5% object with 0.00% background leak and is
flat from 0.97 down to 0.90.

## 6.2 The training loss (rung8, rung9)

```
m     = live_render_mask ∧ gt_mask                      ~28,264 px = 10.8% of 512²

mse   = Σ_{c,i,j} (render − gt)² · m  /  (NORM_PX · 3 + 1e-8)
        NORM_PX = 268,003                                (--loss-norm v4compat)

r     = render·m + (1−m)        composite outside onto white
g     = gt·m     + (1−m)
lpips = LPIPS_alex(2r − 1, 2g − 1)

loss  = mse + 0.1 · lpips  +  λ_reg · mean((raw − frozen)²)
                                     λ_reg = 0.0
```

Backward:

```
(loss · 4096).backward()          LOSS_SCALE=4096 — fp16 in dec_mesh flushes
for p: p.grad /= 4096              small gradients to zero without it
clip_grad_norm_(params, 1.0)
Adam(lr=1e-4, eps=1e-16, weight_decay=0)
```

**Why `NORM_PX` is a constant and not `|m|`.** v4 divided by
`|static_rmask ∪ leaky_gt_mask| ≈ 268,003` while only ~34,000 px actually carried
gradient. A proper mean over the intersection (~28,264) is 9.5× larger for the
same per-pixel error, which at the same LR and grad clip would change the
effective step size — a second variable. `v4compat` keeps the gradient scale
fixed so the intersection change is a true one-variable ablation.

**Why the intersection and not the union.** TRELLIS generates its own teapot; the
Kling video has a different one. Silhouette IoU is **0.7594**. Under a union loss,
**18.2%** of the rendered teapot's pixels sit on white GT background and get
supervised toward white. Residual there is (dark lava − white), the largest
possible, so that thin rim supplied ~56% of the entire colour gradient with sign
`[+,+,+]`. The adapter reads voxel *features*, not position, so it cannot learn
"be white only at the rim" — it learns "be whiter" and the whole body washes out.

Least-squares prediction, modelling the adapter as a global additive shift:

```
union         d* = [Σ_A(g−f) + Σ_B(1−f)] / (|A|+|B|)
intersection  d* =  Σ_A(g−f) / |A|
```

Against v4's actual output over 150 frames: union error 0.027±0.014, intersection
error 0.111±0.019, union closer on 149/150 frames, paired t = 40.0. **v4 did
exactly what the union region demanded.** rung8 then confirmed the fix: dB fell
0.238 → 0.102 with dG landing at 0.087 against a predicted 0.085.

## 6.3 The regularizers that exist but have NEVER run

`cube2mesh.py:158-163`, all behind `if training:`:

```python
reg_loss  = con_loss                             # from sparse_cube2verts
reg_loss += L_dev.mean() * 0.5                   # if mesh.success
reg_loss += (weights[:, :20]).abs().mean() * 0.2
```

| term | penalises | reaches our params? |
|---|---|---|
| `con_loss` | neighbouring cubes disagreeing about a shared corner | **yes** — this is the blockiness fix |
| `0.5 · L_dev` | dual vertex drifting off its edge crossing; stops spikes and self-intersections | **yes** — depends on sdf |
| `0.2 · \|weights\|` | L1 on FlexiCubes α/β, channels `[32:53]` | **no** — we leave `[32:53]` frozen; constant, zero gradient |

## 6.4 Every metric

| metric | definition | notes |
|---|---|---|
| `PSNR_union` | `10·log₁₀(1/MSE)` over `static_rmask` | **v4's region.** Kept unchanged so numbers stay comparable across runs. rung8/rung9 optimise a *different* region, so a lower number here is expected, not a regression |
| `PSNR_inter` | same, over `live_rm ∧ gt_mask` | easier region — never compare to v4 |
| `SSIM` | full image, `channel_axis=2`, `data_range=1` | |
| `LPIPS` | AlexNet, on the union region with v4's leaky mask | parity with v4 |
| `IoU` | `\|rm ∧ gm\| / \|rm ∨ gm\|` | silhouette agreement. Frozen = 0.76 |
| `dRGB` | `mean(lora − frozen)` per channel over the frozen object mask | the white-cast guard |
| **`tracking_ratio`** | `spread%(render sil area) / spread%(GT sil area)`, where `spread% = (max−min)/mean · 100` | **the success criterion for geometry** |

---

# PART 7 — WHY TEXTURE WORKS AND GEOMETRY DOESN'T

Two independent structural reasons. Fixing one does not fix the other.

## 7.1 Texture is pointwise. Geometry is coupled.

```
colour at voxel v  →  affects only the pixels v covers.
                      Change it, nothing else moves.

sdf at cube c      →  moves a surface that c SHARES with 8 neighbours.
                      Change it alone and the surface tears.
```

A LoRA at `out_layer` is **pointwise by construction** — `delta_v = B·A·x_v`,
each voxel independent. Exactly right for colour. Exactly wrong for geometry.

rung9 did not fail because `[0:32]` are the wrong channels. It failed because a
pointwise operator was asked to produce a coupled quantity, with the coupling
term switched off.

> Any geometry edit needs neighbours to agree. Either the **operator** couples
> them (attention, conv — but those leak, see Part 9) or the **loss** couples
> them (`con_loss`). rung9 had neither.

## 7.2 From one camera, texture is observable. Shape is not.

```
every visible surface point  →  has a pixel  →  colour is FULLY determined
the depth of that point      →  no pixel     →  shape is UNDETERMINED
```

The only shape cue in a single view is the **silhouette**. And the intersection
mask removes exactly the pixels where the two silhouettes disagree — the only
place a global shape error is visible.

> The loss that made texture work is the same loss that makes geometry
> impossible. That is structural, not bad luck.

## 7.3 The shape signal is already there — it comes out as colour

Measured on the SLaT cache, CPU only:

- the latent drifts **1.46σ** from frame 1 to frame 150
- temporal variation is **52%** of spatial variation

So the shape change *is* encoded and *does* reach the decoder. The frozen decoder
routes it into appearance.

rung9's evidence: given geometry channels, the adapter put **more** capacity into
them than into colour (`‖B‖geom` 1.305 vs `‖B‖col` 0.862, still climbing at epoch
30) — because geometry was the more efficient *appearance* knob under this loss.

> You do not have a missing-signal problem. You have a routing problem.

---

# PART 8 — EVERY RUN, WITH NUMBERS

## 8.1 Adapter runs

| | v4 `c85c888f` | rung8 `1b0d1d64` | rung9 `e44d8696` |
|---|---|---|---|
| hook | `out_layer` | `out_layer` | `out_layer` |
| channels written | `[53:101]` | `[53:101]` | `[0:32]` + `[53:101]` |
| `B` shape | 48×4 | 48×4 | **80×4** |
| params | 576 | 576 | **704** |
| loss region | union | intersection | intersection |
| render mask | static `mask.png` | **live per frame** | live per frame |
| `λ_reg` | 0.01 — **dead, see 9.1** | 0.0 (honest) | 0.0 |
| PSNR_all (v4 region) | 10.647 | 8.473 | 8.277 |
| SSIM | 0.8671 | 0.8738 | 0.8587 |
| LPIPS | 0.1256 | 0.1355 | 0.1811 |
| IoU | — | — | 0.7097 |
| dR / dG / dB | +0.2457 / +0.2233 / **+0.2380** | +0.1296 / +0.0872 / **+0.1021** | +0.1308 / +0.0837 / **+0.1136** |
| tracking ratio | 0.063 (cannot move geometry) | 0.063 (cannot move geometry) | **0.0983** |

**Read the PSNR column carefully.** It is measured on *v4's* union region. rung8
and rung9 optimise the intersection, so scoring lower there is expected. The
meaningful comparison is rung8 → rung9: **8.473 → 8.277, i.e. 0.2 dB.** rung9
cost almost nothing in image quality.

## 8.2 Diagnostics

| what | result |
|---|---|
| **frozen decoder, pinned coords** — does the SLaT alone give time-varying geometry? | tracking **0.0633**. render spread 0.334% vs GT 5.273%. Max chamfer 0.00526. → *barely* |
| **frozen decoder, free coords** — does stage B track shape if you let it resample? | tracking **1.153**. render spread 6.084% vs GT 5.278%. → *yes, it tracks* |
| **pinned vs free flicker** (both at seed 42) | pinned jump **5.53 px/frame**, free **73.34 px/frame** → **13.3×**. Free vertex count swings 212,648–219,914 |
| **seed fixed vs changing** (stock TRELLIS, 150 frames) | fixed jitter 0.00794 (ratio 33.11), changing 0.01186 (ratio 46.98). IoU 0.991 vs 0.976. → **flicker remains with the seed pinned; the source is per-frame conditioning, not the dice** |
| **rung9 vs frozen** (epoch-12 ckpt, the `lora_best`) | silhouette area spread 0.435% → 0.570%; **vertex spread 1.06% → 11.25%**; mean area 34,477 → 40,723 = **1.18× inflation**; `corr(verts, frame index) = +0.977` — **monotonic drift, not GT-driven oscillation** |
| **v4 turntable** | edit is uniform at every viewing angle — the "texture pasting" hypothesis is **refuted** |
| **rung9 off-view spread** | 0° 0.52%, 90° 0.37%, 180° 0.19%, 270° 0.24% — deformation concentrated in the supervised view |

## 8.3 What rung9's failure decomposes into

| symptom | cause | fix |
|---|---|---|
| blocky, faceted surface | per-cube sdf with no consistency term; `con_loss` never computed because `dec_model.eval()` | `training=True` on the extractor + `λ_con · reg_loss` |
| tracking 0.098, not 1.0 | loss cannot see shape — one view, intersection-masked, interior RGB only | silhouette term `\|render_mask − gt_mask\|` |
| **1.18× inflation** | **unattributed.** Staircase edges add ~2–4% of area at this resolution, not 18%, so most is real inflation — but whether it comes from incoherence or from the optimiser is unknown | the test that separates them: rerun with `con_loss` on and nothing else changed |

---

# PART 9 — BUGS FOUND (the audit trail)

## 9.1 v4's regularizer was dead — inherited bug

```python
frozen_col = new_feats[:, C0:C1]      # a VIEW
...
new_feats[:, C0:C1] = raw_col.clamp(...)   # overwrites what frozen_col points at
```

So the regulariser computed `(raw − clamp(raw)) ≈ 0`. Confirmed in v4's
`loss_history`: reg ~1e-13 against mse ~1.7e-2, all 30 epochs. **v4's config says
`λ_reg=0.01` but v4 is an unregularized run.** Matters for the writeup.

Fixed in rung8 with `.clone()`. rung8/rung9 set `λ_reg=0.0` explicitly so they
honestly reproduce v4's *effective* behaviour.

## 9.2 GT mask was leaky

`(gt < 0.99).any()` returned 99.7% of the image because the background is
off-white. Would have made the intersection loss a **no-op**. Fixed → `min(RGB) <
0.95`.

## 9.3 float16 underflow in visibility scores

`grad/4096` flushes to zero below 6e-8, silently turning a soft mask hard.
Fixed → float32.

## 9.4 `set -o pipefail` under dash

`sbatch --wrap` runs `/bin/sh` = dash. Jobs 2143676/2143677 died in 0 s. All jobs
are now real `.sbatch` files with `#!/bin/bash`.

## 9.5 Stride-5 subsampling of a consecutive-frame metric

Flicker is by definition a frame-to-frame quantity. Measuring every 5th frame
measures 5-frame drift instead. Invalid metric, fixed to stride 1.

## 9.6 Mislabelled seed arms

Called `seed = frame index` "stock TRELLIS". Stock defaults to `seed=42`. Job
2147822 cancelled and redesigned.

## 9.7 Seed confound between panels

`render_pinned_vs_free.py` compared a seed-42 cache against a seed-6 live sample.
Two variables differed. Job 2148107 killed, resubmitted as 2148109 with both arms
at 42. Ratio came down 19.9× → 13.3× — some of the earlier number *was* the seed.

## 9.8 Provenance was never asserted (found in the rung9 audit)

Nothing checked that the SLaT cache's `coords` actually came from frame 75 at
seed 42 — only that the **count** was 7,301. This is 9.7's root cause. rung9 now
asserts voxel-for-voxel equality against a fresh sample. *That check has not yet
run on any completed job.*

## 9.9 GATE-geom perturbation inside the fp16 floor

Probed the geometry channels at σ=1e-3. The decoder runs fp16, where `1.0 + 1e-4`
rounds back to `1.0`. Could have aborted the job for a dtype reason, not a real
one. Now sweeps σ ∈ {1e-3 … 1.0} and reports the floor.

## 9.10 Wrong comparator for the per-epoch tracking ratio

Printed against a hardcoded 0.063 that came from a *different frame set* (stride
10 vs stride 15). Now measures the frozen ratio on the same frames and persists
it across requeue.

## 9.11 Placement leaks (rung5.5 and earlier)

`rung5_55_r16` put rank-16 adapters on `attn.to_qkv`, `attn.to_out`,
`mlp.mlp[0]`, `mlp.mlp[2]` of decoder blocks 8–11 — 786,432 params. Three mixing
stages sit downstream:

| stage | effect on a per-voxel edit |
|---|---|
| windowed attention, blocks 8–11 | spreads it across an 8³ window, shifted between blocks |
| `upsample[0]` 3×3×3 conv | bleeds one voxel at res 128 |
| `upsample[1]` 3×3×3 conv | bleeds one more at res 256 |

The edit lands on voxels adjacent to the surface. Those are outside the
isosurface, so they get colour but the surface never passes through them —
a faint shell hanging off the object. **This is why the hook moved to
`out_layer`.**

---

# PART 10 — THE TWO ROUTES

| | **route A — learn the deformation** | **route B — let TRELLIS do it** |
|---|---|---|
| how | pin `coords`, adapter moves the surface | free `coords`, stage B resamples per frame |
| tested by | rung9 | `geom_free`, `pin_vs_free` |
| shape tracking | **0.098** (frozen 0.063) | **1.153** — already correct |
| temporal coherence | **5.53 px/frame** — excellent | **73.34 px/frame** — 13.3× worse |
| what's broken | pointwise operator, shape-blind loss | discrete topology jumps between frames |
| fixes needed | 2, both identified | 1, not yet identified |

The two routes have **complementary** failures. A has coherence and no tracking;
B has tracking and no coherence.

FlexiCubes documents B's failure as fundamental: when the isosurface slips over a
grid vertex, the mesh jumps discontinuously. The vertex count swinging 7,266
across frames is exactly that.

**Recommendation: route A.** Both of its failures have named causes and named
fixes. B's does not.

---

# PART 11 — WHAT rung10 WOULD BE

One variable on top of rung9.

| # | Where | rung9 | rung10 |
|---|---|---|---|
| 1 | `geomcol_forward()` | no `training` arg | add one; set `dec_model.training = flag` around the forward, restore after. **Attribute only** — `dec_model.train()` recurses into all 12 blocks |
| 2 | training forward | `training=False` | `training=True` — turns `con_loss` on |
| 3 | loss | `mse + 0.1·lpips` | `+ λ_con · mesh.reg_loss` |
| 4 | every eval path | eval | **stays eval** — so PSNR/IoU/tracking/vertex counts remain comparable |
| 5 | new arg | — | `--lambda-con` |
| 6 | new logging | — | `con_loss` and `L_dev` **separately** (the third term is a constant) |
| 7 | new gate | — | **GATE-con**: assert `reg_loss > 0`. If it's 0.0 the flag didn't reach the extractor and rung10 is rung9 again |

Unchanged: channels, intersection loss, pinned coords, rank 4, LR 1e-4, seed 6,
30 epochs, LOSS_SCALE 4096.

**λ_con is unknown.** `reg_loss` has never been evaluated in any run. `mse` is
~0.007. A 2-minute probe job prints its magnitude, then pick λ so the term starts
at 20–30% of `mse`.

rung11 would add the silhouette term. The renderer already antialiases the mask
(`mesh_renderer.py:107`), so `|render_mask − gt_mask|` is differentiable w.r.t.
vertex positions.

---

# PART 12 — OPEN QUESTIONS

1. **λ_con** — never measured.
2. **The 1.18× inflation** — cause unattributed. rung10 settles it.
3. **Albedo-only has never been tested.** Every run writes all 48 appearance
   channels — 24 albedo *and* 24 shading normals. `--delta-channels rgb` exists in
   several scripts and no run has used it.
4. **rung8 has no turntable.** We know it fixes the white cast head-on; not
   whether the fix holds off-view.
5. **No adapter is time-conditioned.** Temporal variation enters only through the
   per-frame SLaT. The paper's α-modulation claim has no implementation.
6. **`[32:53]` never tested.** `--write-blocks all101` exists in rung9 and has
   never been run.
7. **Cache provenance unverified** for every completed run (see 9.8).

---

# PART 13 — FILE MAP

| file | what |
|---|---|
| `lora_experiments/rung5_colonly_lora_v4.py` | v4 — colour-only, union loss |
| `lora_experiments/visibility/rung8_intersection_lora.py` | rung8 — intersection loss |
| `lora_experiments/visibility/rung9_geomcolor_lora.py` | rung9 — + geometry channels |
| `lora_experiments/visibility/render_rung9_comparison.py` | GT │ frozen │ rung9 video; reads block layout from the ckpt |
| `lora_experiments/visibility/measure_geometry_change.py` | tracking ratio, frozen decoder, pinned coords |
| `lora_experiments/visibility/geom_change_free_coords.py` | same, free coords |
| `lora_experiments/visibility/render_pinned_vs_free.py` | pinned vs free video |
| `lora_experiments/visibility/seed_fixed_vs_changing.py` | seed ablation |
| `trellis/representations/mesh/cube2mesh.py` | channel layout, `reg_loss` assembly |
| `trellis/representations/mesh/utils_cube.py` | `cubes_to_verts` averaging, `con_loss`, `get_defomed_verts` |
| `trellis/representations/mesh/flexicubes/flexicubes.py` | `_triangulate`, the training-mode centre vertex |
| `trellis/models/structured_latent_vae/decoder_mesh.py` | decoder forward, `training=self.training` at line 158 |
| `trellis/renderers/mesh_renderer.py` | `dr.antialias` on colour and mask |
| `experiments/EXPERIMENT_LOG.md` | running job tracker |
