# Citation audit v2 — FULL TEXT

**Date:** 2026-07-29
**Target:** `dynamesh_intro_related_work.md` (the original artifact export)
**Method:** fetched `arxiv.org/html/<id>v1` for every reference with an arXiv ID and
searched the full text — Related Work, Method, Experiments — for each specific
claim the draft makes. Classics without IDs were checked via their arXiv abstract
page for title and venue.

---

## Why this audit supersedes the first one

The first audit (`CITATION_AUDIT.md`) judged claims from **abstracts only**. That
produced a high false-positive rate, because abstracts systematically omit which
backbone a paper builds on — that information lives in Section 3 or 4.

**Four of the first audit's findings were wrong, including both of its two
"critical" ones.** The original draft was substantially more accurate than the
audit claimed. Specifically, these were flagged as fabricated and are in fact
correct:

| Claim flagged as wrong | Full text says |
|---|---|
| FlowBender fine-tunes TRELLIS.2's texture transformer with LoRA | *"fine-tuning the TRELLIS-2 texture transformer"* + *"integrate LoRA adapters into all linear layers"* |
| Steer3D trains adapters on TRELLIS's two stages | *"We choose TRELLIS … as our base model"* + *"TRELLIS contains two separate flow models for geometry and texture … we perform separate training for each stage"* |
| TRELLIS.2's "9,600 tokens / 16× downsampling" unsourced | *"only ∼9.6K latent tokens"* + *"16× spatial downsampling"* + *"Sparse Compression VAE (SC-VAE)"* |
| MorphAny3D does not name TRELLIS | *"we adopt the image-to-3D variant of Trellis"* |

**Lesson for this project: do not audit citations from abstracts.**

---

## Verdicts — claims WITH an arXiv ID

| Ref | Paper | Names TRELLIS? | Draft's claim |
|---|---|---|---|
| [8] | TRELLIS 2412.01506 | is TRELLIS | **OK** — two-stage design and SLAT verified directly against this repository's code and configs, which is stronger evidence than the paper text |
| [11] | TRELLIS.2 2512.14692 | as prior work | **OK** — 9.6K tokens, 16× downsampling, SC-VAE all confirmed verbatim. Two nits: the term is **O-Voxel** (singular), and the paper positions itself as a *new native representation* rather than "extending the original framework" |
| [12] | UniLat3D 2509.25079 | **YES** | **MOSTLY OK** — two-stage critique, geometry–texture misalignment, and unified geometry-appearance VAE all confirmed. **Overstated:** the paper does not argue the appearance stage *cannot correct* geometric errors; it argues misalignment arises. Soften |
| [13] | SpaceControl 2512.05343 | **YES** — *"such as Trellis"* | **CLAIM OK, ATTRIBUTION WRONG** — superquadrics and the blending parameter confirmed. **Affiliations are ETH Zurich, Stanford, Technion, NVIDIA — NOT Microsoft Research.** Also the paper does not use the term "SLaT" |
| [14] | Steer3D 2512.13678 | **YES** | **CLAIM OK, TITLE WRONG** — ControlNet-style adapters trained separately per stage confirmed. Real title: *"Feedforward 3D Editing via Text-Steerable Image-to-3D"*; there is no "Steer3D:" prefix. Year 2024 → Dec 2025 |
| [15] | StyleSculptor 2509.13301 | **YES** — *"we adopt TRELLIS … as the backbone"* | **OK** — SD-Attn confirmed; geometry-only and texture-only guidance confirmed. Nit: the control is a channel-count hyperparameter *K*, not literally "separately extracts" two feature sets |
| [16] | FlowBender 2606.20404 | **YES** — TRELLIS-2 | **OK — RESTORE THIS CITATION.** LoRA adapters and texture-transformer fine-tuning both confirmed verbatim. This is the closest prior work to DynaMesh |
| [17] | MORPHOS 2606.02491 | **YES** | **OK** — T-SLat, *"two autoregressive flow models"*, evolving topologies all confirmed |
| [18] | SS4D 2512.14284 | **YES** — *"builds upon TRELLIS"* | **OK** — factorized 4D convolutions, temporal layers, 3D Gaussian sequences confirmed. Additional detail worth using: it *fine-tunes TRELLIS's autoencoder and generator* and curates **16,000 animated objects** — concrete evidence for our "requires 4D data" contrast |
| [19] | Helix4D 2605.26109 | **YES** — Trellis2 | **OK** — sliding-window cross-frame attention, first-frame anchor, RoPE-band temporal encoding confirmed |
| [20] | ArtiLatent 2510.21432 | **YES** | **OK** — unified VAE and articulation-conditioned appearance decoding confirmed. **Note: outputs 3D Gaussians only, not meshes.** Do not cite as mesh-side precedent |
| [21] | T2Mo 2606.05162 | **NO** | **ONE CLAIM WRONG.** "T2Mo" *is* the authors' own abbreviation — the first audit was wrong to call it invented. Trajectories + text and the motion/appearance separation are confirmed. **But "generating geometry and texture jointly" is FALSE — it generates per-vertex displacements for a given static mesh.** Fix or drop |
| [22] | MorphAny3D 2601.00204 | **YES** | **OK** — training-free, SLAT feature aggregation inside attention, MCA and TFSA confirmed |
| [23] | Interp3D 2601.14103 | **YES** | **OK** — three phases confirmed as *Semantic-Aligned Condition Interpolation*, *SLAT-Guided Structure Interpolation*, *Fine-Grained Texture Fusion*. **ICLR 2026 is not stated anywhere in the paper** |
| [24] | Wukong 2511.22425 | **YES** | **OK** — Trellis backbone, free-support Wasserstein barycenter, training-free confirmed. **NeurIPS 2025 is not stated anywhere in the paper** |

## Verdicts — classics, checked via arXiv page

| Ref | Verified | Issue |
|---|---|---|
| [1] NeRF `2003.08934` | ECCV 2020 (oral) | none |
| [2] 3D Gaussian Splatting `2308.04079` | ACM TOG 42(4), 2023 | none (TOG 42(4) *is* SIGGRAPH 2023) |
| [3] DreamFusion `2209.14988` | arXiv Sep 2022 | ICLR 2023 **not stated on the arXiv page** — widely reported, confirm |
| [4] Magic3D `2211.10440` | CVPR 2023 | none |
| [5] LRM `2311.04400` | ICLR 2024 | none |
| [6] InstantMesh `2404.07191` | arXiv Apr 2024 | none |
| [7] CraftsMan `2405.14979` | arXiv May 2024 | **Title is "CraftsMan3D"**, and **NeurIPS 2024 is not stated** |
| [9] DINOv2 `2304.07193` | arXiv Apr 2023 | TMLR 2024 **not stated** — confirm |
| [10] FlexiCubes `2308.05371` | ACM TOG 42(4), 2023 | **Title is "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization"** — FlexiCubes is the method name, not in the title |

## Added

| Ref | Paper |
|---|---|
| [25] | Dinh, Lang, Stein, Hanocka. **"RADmesh: Remesh-Aware Mesh Deformation."** ECCV 2026 (Oral). Text-guided *localized* deformation with adaptive remeshing, optimized through DeepFloyd IF on a coarse-to-fine remeshing schedule. Input: mesh + boolean vertex-selection + prompt |

**Repo bug worth reporting to the authors:** the BibTeX in
`github.com/threedle/radmesh` README has the wrong `title` field —
*"Improving 2D Feature Representations by 3D-Aware Fine-Tuning"* (a different
paper). Key and authors are right; anyone citing from that README cites it wrong.

---

## The actual error list — 9 items

1. **[13]** remove "Microsoft Research" → ETH Zurich, Stanford, Technion, NVIDIA. Year 2024 → 2025
2. **[14]** title → *"Feedforward 3D Editing via Text-Steerable Image-to-3D"*. Year 2024 → 2025
3. **[21]** *"generating geometry and texture jointly"* is false — per-vertex displacement on a given static mesh. Also does not name TRELLIS
4. **[7]** title → *"CraftsMan3D"*; drop the NeurIPS 2024 claim
5. **[10]** title → *"Flexible Isosurface Extraction for Gradient-Based Mesh Optimization"*
6. **[23]** drop "ICLR 2026"
7. **[24]** drop "NeurIPS 2025"
8. **[12]** soften "cannot correct geometric errors"
9. **[11]** "extends the original framework" → positions itself as a new representation; term is O-Voxel

Everything else in the original draft's TRELLIS-related claims checks out against
full text. **12 of 15 arXiv-ID claims were correct as written.**

## Still not verified

**Author lists.** Confirmed: MORPHOS (Kwon, Choi, Shin, Kim, Lee, Kim), Text2Mesh
(Michel, Bar-On, Liu, Benaim, Hanocka), Geometry in Style and RADmesh (Dinh, Lang,
Stein, Hanocka — RADmesh adds no fifth author), SpaceControl affiliations. All
other "X et al." attributions are unchecked.

**Venues** for DreamFusion (ICLR 2023), DINOv2 (TMLR 2024), CraftsMan3D,
InstantMesh — confirm against official proceedings.
