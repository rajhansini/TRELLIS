# Experiment log — DynaMesh

Running record so nothing gets lost. Newest at the bottom of each section.
Updated 2026-07-30.

---

## IN FLIGHT

| Job | Name | Question it answers | Output | Status |
|---|---|---|---|---|
| **2147831** | `seed_test` | Does pinning the random seed remove flicker? Fixed seed=6 vs seed=frame-index, stock TRELLIS, 150 frames, one argument apart. | `visibility/seed_test/SEED_fixed_vs_changing.mp4` — GT \| fixed \| changing | PENDING |
| **2147837** | `geom_free` | Does freeing the coords fix geometry tracking? Re-measures the 0.063 ratio with per-frame sparse structure instead of pinned frame-75. | `visibility/diagnostics/geomchange_free_coords/` | PENDING |

**How to read them**

- `seed_test` → if fixed is smooth and changing jitters, the seed buys your stability. If both jitter, the seed is not the source — per-frame conditioning is, and only pinning coords removes it. **Watch the vertex-count panel**: if it swings even with the seed fixed, the sparse structure is driven by the image, not the dice.
- `geom_free` → tracking ratio near 1.0 means stage B already does the shape and **no geometry LoRA is needed**. Near 0.0 means freeing coords does not help and the LoRA is justified. Also reports a flicker ratio, since free coords is expected to cost coherence.

---

## DONE

| Job | Name | Question | Answer |
|---|---|---|---|
| 2146409 | `v4_360` | Does v4's texture paste onto surfaces the camera never saw? | **No.** The edit is uniform at every angle. But the frozen baseline already wraps lava correctly all round, and v4 degraded it from every viewpoint — pre-rung8, the adapter was worse than doing nothing. |
| 2146410 | `rung8_isect` | Does restricting the loss to the render∩GT intersection remove the white cast? | **Yes, CONFIRMED.** ΔB 0.238 → 0.105, past the 0.16 threshold set in advance. ΔG landed at 0.086 vs a predicted 0.085. `clip_frac` 0.0000, so it stayed a true one-variable ablation. |
| 2146411 | `geom_change` | Does the frozen decoder give time-varying geometry? | **Barely.** Tracking ratio **0.063** — TRELLIS's silhouette moves 0.30%, the GT's moves 5.43%. Chamfer rises to 0.005 by frame 20 then goes flat: one adjustment early, then frozen. Measured with **coords pinned to frame 75** — which is what `geom_free` is now re-testing. |
| 2147686 | `final_cmp` | Side-by-side of everything. | `visibility/final_comparison/FINAL_gt_frozen_v4_rung8.mp4`. Mean RGB at frame 75: GT [118.6, 58.1, 34.9] · frozen [79.3, 28.9, 8.4] · v4 [136.3, 84.6, 73.0] · **rung8 [117.0, 51.1, 35.2]**. |

**CPU-only, no job:** the SLaT latent drifts **1.46σ** from frame 1 to 150, and temporal variation is 52% of spatial variation. So the shape signal *does* reach the decoder — the decoder routes it into appearance and not geometry.

---

## CANCELLED — and why

| Job | Why killed |
|---|---|
| 2143676, 2143677 | `set -o pipefail` under `sbatch --wrap`, which runs `/bin/sh` = dash. Died in 0 s. Fixed by moving to real `.sbatch` files with a bash shebang. |
| 2147822 | Two-arm design where I labelled `seed=frame index` as "stock TRELLIS". Stock already defaults to `seed=42`, so the arms were mislabelled and it wasn't the experiment that was asked for. |
| 2148107 | `pin_vs_free` with the seed unmatched across panels. PINNED reads `slat_cache.npz` (built at `STRUCT_SEED=42`); I set the FREE arm's default to 6 without checking the cache. Two variables differed, so no shape difference could be attributed to coords. Resubmitted as **2148109** with both arms at seed 42. |

---

## WRITTEN BUT NEVER RUN

| Script | What it does | Blocker |
|---|---|---|
| `rung9_geomcolor_lora.py` + `jobs/rung9.sbatch` | **rung8 with the LoRA write window widened to geometry.** `[0:32]` sdf+deform **and** `[53:101]` colour; `B` 80×4 instead of 48×4. Everything else byte-identical to rung8 — loss, intersection mask, pinned frame-75 coords, rank 4, LR 1e-4, seed 6, 30 epochs, LOSS_SCALE 4096. Success = tracking_ratio > 0.20 (frozen baseline **0.063**), logged every epoch. Guard = dB must stay near rung8's +0.105, not climb to v4's +0.238. `--write-blocks color` reproduces rung8 exactly (576 params); `all101` adds `[32:53]` for the full-out_layer ablation. | Written and verified, **not submitted** |
| `compute_visibility_mask.py` | Per-voxel gradient probe: which voxels the training camera ever saw | Never submitted — fell off the list during the `pipefail` rebuild. Its premise is also weakened: the turntable showed v4's edit is uniform at all angles, not concentrated where the camera looked. |
| `rung7_vismask_lora.py` | Retrain with the visibility mask baked in | Needs the mask above |
| `analyze_texture_wrapping.py` | Is the wrap driven by feature similarity? | Needs the mask above |
| `diagnose_geometry_vs_appearance.py` | Is the artifact geometry or appearance? | Superseded — `geom_change` answered it |

---

## OPEN QUESTIONS

1. **Does freeing the coords cost more than it buys?** `geom_free` gives the shape half. Its flicker number gives the other half.
2. **Is a geometry LoRA needed at all?** Depends entirely on `geom_free`'s tracking ratio.
3. **Albedo-only has never been tested.** Every run writes all 48 appearance channels — 24 albedo **and 24 shading normals**. `--delta-channels rgb` exists in several scripts but not in `rung8_intersection_lora.py`, and no run has used it.
4. **rung8 has no turntable.** We know it fixes the white cast head-on; we do not know whether the fix holds off-view.
5. **No adapter is time-conditioned.** Temporal variation enters only through the per-frame SLaT. The paper's α claim has no implementation.

---

## REFERENCE — where the numbers came from

| Fact | Source |
|---|---|
| out_layer channel split: sdf `[0:8]`, deform `[8:32]`, weights `[32:53]`, colour `[53:101]` | `trellis/representations/mesh/cube2mesh.py` LAYOUTS |
| colour block is 8 corners × 6 = 24 albedo + 24 shading normal | same, `"[4:7] color [7:10] normal"` |
| Vertex positions read only sdf and deform | `cube2mesh.py`, `get_defomed_verts(reg_v, deform_d, res)` |
| out_layer is the last op that does not mix voxels | 12 windowed-attention blocks and 2 conv upsamplers precede it |
| 7,301 coarse → 467,264 fine voxels | two `SparseSubdivideBlock3d`, 8× each |
| Silhouette IoU render vs GT = 0.76, 16% of render on GT background | measured across 150 frames |
| v4's `lambda_reg=0.01` was a no-op | tensor-view aliasing; `loss_reg` ~1e-13 vs `loss_mse` ~1.7e-2 for all 30 epochs |
| Stock TRELLIS defaults to `seed=42` | `trellis_image_to_3d.py`, `run()` signature |
