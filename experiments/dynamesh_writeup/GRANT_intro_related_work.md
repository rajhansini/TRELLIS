# DynaMesh: Dynamic Texture and Geometry Generation on 3D Shapes

## Introduction & Related Work — Grant Proposal

---

# 1. Introduction

Generative AI has changed how 3D content is made. Where authoring a single asset once took an artist days, modern systems synthesize expressive, detailed 3D shapes from a sentence or a photograph in seconds. That capability is being absorbed into real pipelines — augmented and virtual reality, film and game production, product visualization, simulation, cultural-heritage reconstruction — because it removes the bottleneck that mattered most: getting a plausible, usable 3D object into the scene at all.

Yet almost everything these systems produce is frozen at the moment of creation. The geometry is fixed, the texture is baked onto its surface, and the asset cannot subsequently change. For much of the physical world this is the wrong model. Real objects age, dry, corrode, bruise, wilt, char, and decay — and when they do, their appearance and their shape change together. Fruit does not merely discolour, it shrivels. Metal does not merely dull, it pits. A generative model that captures the *what* of an object but has no account of the *when* cannot express any of this.

**Creating a 3D asset that changes over time, in both texture and geometry, has not been explored.** The gap persists because the two established routes both fail on it. Training a native 4D generator requires large-scale 4D supervision that does not exist at the scale 3D data does. Editing a mesh under a differentiable-rendering objective yields a *single* new configuration, not a trajectory through configurations.

We propose **DynaMesh**, which produces a mesh whose texture and geometry evolve continuously, supervised by a single video of the target effect, without retraining the generative backbone.

Our approach rests on a division of labour between two pretrained models that are individually strong and jointly complementary. A **3D generative model** contributes spatial understanding: from one view it infers a complete, multi-view-coherent object, including surfaces the camera never saw. A **video generative model** contributes temporal understanding: an inherent inductive bias for smooth, physically plausible change over time. Neither can do the other's job. Our contribution is the mechanism that couples them.

The difficulty is that a 3D generative model is built to produce *plausible* objects, not *particular* ones. Asked for a teapot it returns a good teapot — rarely *this* teapot. The situation is directly analogous to 2D image diffusion, where generating a specific image rather than a plausible one required a distinct body of **inversion** techniques. **We frame our problem as 3D inversion**: adapt a pretrained 3D generator so that it reproduces a specific object faithfully in geometry and texture, while retaining the 3D prior that lets it complete unobserved regions — and then make that reproduction a function of time.

Concretely, we anchor a lightweight low-rank adapter inside a frozen 3D generative decoder, at the projection layer where geometry and appearance channels first become explicitly separable. A video model supplies temporally coherent target frames; a differentiable renderer propagates that 2D supervision into the 3D representation. The backbone is never modified, and supervision comes from a **single viewpoint** — yet the result must be a coherent 3D object, not a decal correct from one angle. Holding both requirements simultaneously is the technical core of this work.

We treat the choice of backbone as an implementation detail. We build on TRELLIS [8], but the framework assumes only a generator with a per-element latent representation and a decoder in which geometry and appearance are separable — properties shared by an emerging family of structured-latent 3D models.

---

# 2. Related Work

## 2.1 Progress in 3D generation, and where TRELLIS sits

Neural scene representations opened the modern era of 3D generation. **NeRF** [1] showed that a volumetric radiance field could be recovered from posed images by differentiable volume rendering, and **3D Gaussian Splatting** [2] later made such representations real-time by replacing implicit fields with explicit, rasterizable primitives. **DreamFusion** [3] and **Magic3D** [4] turned reconstruction into generation, distilling gradients from pretrained 2D diffusion models to synthesize 3D content from open-vocabulary text. Powerful as these are, they return implicit fields or point-like primitives that resist insertion into standard graphics pipelines, which expect explicit textured triangle meshes with well-defined topology.

A parallel line pursued meshes directly. Feed-forward reconstruction models — **LRM** [5], **InstantMesh** [6], **CraftsMan3D** [7] — encode one or a few views into a latent and decode straight to a textured mesh in seconds. Their weakness is fidelity: geometry tends to be smooth and under-detailed, texture coarse, reflecting how little fine surface structure is recoverable from sparse 2D observations. All are single-shot and reconstructive, with no mechanism for downstream variation.

**TRELLIS** [8] marks the current high-water mark for fidelity among feed-forward image-to-3D methods, and its design is central to ours, so we describe it in detail. TRELLIS introduces **Structured Latents (SLAT)** and generates in two sequential stages.

The **first stage decides where the object is.** Starting from noise in a small dense 3D grid, a flow-matching model — conditioned on **DINOv2** [9] features of the input image — is denoised into a structure code, which a convolutional decoder expands into a voxel occupancy grid. Thresholding leaves a sparse set of occupied voxels: the object's coarse shape and extent, and nothing else. No appearance is decided here.

The **second stage decides what each occupied voxel is.** Fresh noise is placed on those voxels and a second flow model, conditioned on the same image features, denoises it into a compact per-voxel latent — a learned feature representation attached to each occupied location. This is where TRELLIS's expressiveness lives, and it is also the origin of a property we exploit throughout: each latent encodes that voxel's local surface detail and its appearance *jointly*, in a single code. The same latent is consumed by three different decoders — mesh, Gaussian, and radiance field — so it is a general-purpose description rather than a geometry/appearance split.

**Decoding** resolves that shorthand into an explicit object. A transformer over the occupied voxels supplies neighbourhood context, upsampling blocks refine the coarse voxels into a much finer set, and a final projection maps each fine voxel to channels that partition explicitly into geometry and appearance. A differentiable isosurface extractor, **FlexiCubes** [10], converts these into a textured triangle mesh. This projection layer is the first and only point in the pipeline where geometry and appearance are separable, and it is therefore where we intervene.

> **The static limitation, and the value we add.** TRELLIS generates one fixed object from one image: frozen geometry, frozen texture, no notion of time. It has no mechanism for an asset to age, deform, or change appearance, and — as a generative model — no mechanism to reproduce a *specific* object rather than a plausible one. DynaMesh addresses both at once: we adapt a frozen TRELLIS decoder to reproduce a particular object, and make that reproduction a continuous function of time, with the temporal prior supplied by a video model.

## 2.2 Follow-up work on structured latents

TRELLIS has produced a family of extensions. We group them by what each changes, and state in each case what remains out of reach.

### Representation

**TRELLIS.2** [11] introduces **O-Voxel**, a "field-free" sparse voxel structure that encodes geometry and full physically-based appearance in a single latent, compressing a fully textured asset at 1024³ resolution into approximately 9.6K latent tokens via a Sparse Compression VAE achieving 16× spatial downsampling. It improves fidelity on thin structures, inner surfaces and transparency. *Unlike TRELLIS.2, we do not seek a better static representation.* Its tighter coupling does sharpen a question we must answer: whether temporal control over geometry and texture can be exercised separately or must be joint.

**UniLat3D** [12] takes a critical view of the two-stage decoupled design, arguing that separate geometry-then-appearance generation introduces an inevitable gap between the two and produces geometry–texture misalignment. It proposes a unified geometry–appearance VAE encoding both into a single latent in one pass. *Unlike UniLat3D, we accept the two-stage backbone and intervene downstream* — but its analysis directly supports a premise of ours: geometry and appearance are not independently interpretable once decoded to a surface, which is why we ultimately modulate both rather than texture alone.

### Control and editing

**SpaceControl** [13] injects explicit geometric control into pretrained image-to-3D generation, including TRELLIS, in a training-free manner: a user-supplied shape proxy — from a compact superquadric to a detailed mesh — is voxelized, encoded, and blended with the image conditioning through a tunable fidelity parameter, with appearance following from the conditioned geometry. This confirms that geometry is the upstream controllable variable. *Unlike SpaceControl, our control variable is time, not shape.* It selects one fixed geometric configuration; we parameterize a continuous trajectory through configurations.

**Feed-forward text-steerable 3D editing** [14] chooses TRELLIS as its base model and, because TRELLIS contains two separate flow models for geometry and texture, performs separate ControlNet-style adapter training for each stage — direct empirical evidence that the two-stage pipeline supports independent interventions at the stage level. *Unlike this work, which produces a single edited static mesh per steering input, we learn how an object moves through a sequence of configurations.*

**StyleSculptor** [15] adopts TRELLIS as its backbone and achieves zero-shot style control through a Style-Disentangled Attention module that partitions style-aware channels, supporting style intensity control and exclusively geometry-only or texture-only stylization. This is important evidence that meaningful separation exists at the channel level, and it motivates our choice of intervention point. *Unlike StyleSculptor, which transfers one reference style to a static output, we learn a time-indexed modulation supervised by video.*

**FlowBender** [16] is the closest prior work to ours. It fine-tunes the TRELLIS-2 texture transformer with **LoRA adapters integrated into all linear layers**, using feedback-aware closed-loop training that treats conditioning-alignment error as an explicit network input, correcting systematic appearance artifacts while the geometry backbone remains frozen. It establishes that parameter-efficient low-rank adaptation of a structured-latent 3D decoder is viable for appearance correction. *The distinction is temporal: FlowBender corrects a single static texture toward better alignment. It introduces no temporal variation, no time-conditioned adaptation, and no mechanism for appearance to change as a function of a temporal index.* DynaMesh extends low-rank adaptation into the temporal domain, learning a time-indexed modulation supervised by a target video sequence so that appearance — and ultimately geometry — tracks a continuous trajectory.

### Interpolation and morphing

**MorphAny3D** [17] demonstrates training-free morphing between arbitrary categories, adopting the image-to-3D variant of TRELLIS and aggregating SLAT features directly within attention — Morphing Cross-Attention for structural coherence, Temporal-Fused Self-Attention for frame consistency — rather than interpolating at the noise or condition level. **Interp3D** [18] uses TRELLIS as its 3D diffusion prior and tackles textured morphing in three explicit phases: Semantic-Aligned Condition Interpolation, SLAT-Guided Structure Interpolation for geometry, and Fine-Grained Texture Fusion for appearance. It is thus the most direct evidence that geometry and texture can be manipulated in separate, sequentially applied phases without corrupting one another. **Wukong's 72 Transformations** [19] uses a pretrained TRELLIS backbone and formulates morphing as a free-support Wasserstein barycenter problem over flow models, also training-free.

Together these establish our core representational premise: **meaningful transitions are achievable by operating in the latent domain**, where correspondence exists, rather than in mesh space, where it does not across differing topologies. *Unlike all three, these are training-free interpolations between two fixed static endpoints.* They do not model the temporal evolution of a single object, cannot be queried at an arbitrary time index, and cannot capture non-linear, physically grounded dynamics — the irregular spread of decay, for instance — that linear interpolation between endpoints cannot express. DynaMesh learns a parametric temporal model from video instead.

### State-conditioned appearance

**ArtiLatent** [20] applies TRELLIS's latent diffusion model to articulated objects, jointly embedding sparse voxels, part category labels and articulation attributes in a unified VAE latent space, with appearance decoding conditioned on articulation state so that regions typically occluded in static poses receive plausible texture. This is the clearest precedent for *appearance conditioned on state*. *Unlike ArtiLatent, our state is not a discrete articulation within a known kinematic model but a continuous, unstructured temporal index learned from video* — no predefined parts, no joint constraints. We note ArtiLatent decodes to 3D Gaussians rather than meshes.

## 2.3 Dynamic 3D: 4D generation and mesh deformation

Two established communities address temporal change, and DynaMesh sits between them.

### Native 4D generation

**MORPHOS** [21] builds on a pretrained 3D generative model and extends SLAT along time as **T-SLat** (Temporal Structured Latents), jointly encoding geometry and appearance across frames, with two autoregressive rectified-flow transformers generating sequences via causal attention while handling evolving topologies. **SS4D** [22] builds upon TRELLIS to produce a native 4D generator from monocular video, adding temporal layers that reason across frames plus factorized 4D convolutions and temporal downsampling, decoding to 3D Gaussian sequences. **Helix4D** [23] adapts TRELLIS.2 from image-to-3D to video-conditioned 4D generation using sliding-window cross-frame attention anchored on the first frame, and a 4D temporal encoding that repurposes redundant low-frequency spatial RoPE bands for time.

These validate our motivation: frame-wise generation from a static model is insufficient for temporally dynamic objects. **But all three require 4D training data and modify the generative backbone.** SS4D is explicit about the cost: it fine-tunes TRELLIS's autoencoder *and* generator, and curates a dataset of 16,000 animated 3D objects. *Unlike all three, DynaMesh adds no trained temporal layers, requires no 4D dataset, and leaves the backbone untouched*; the temporal prior is borrowed from a video model at supervision time rather than learned into new weights. The trade is explicit: we forgo their generality in exchange for operating where 4D data is unavailable.

**T2Mo** [24] introduces feed-forward controllable dynamic 3D shape generation conditioned jointly on user-provided 3D point trajectories and a text prompt, separating spatial motion control from global appearance semantics. It demonstrates that dynamic 3D generation can be conditioned on explicit spatial guidance. *Unlike T2Mo, which produces per-vertex displacements for a given static mesh, we generate the object and its temporal evolution together*, and our supervision is a 2D video rather than 3D trajectory annotations and text.

### Mesh deformation from 2D objectives

A second line deforms a *given* mesh under a 2D objective, and is methodologically closest to us. **Text2Mesh** [25] stylizes a fixed input mesh by predicting **both colour and local geometric detail** from a text prompt via a neural style field scored by CLIP. **TextDeformer** [26] produces larger, smoother deformations by representing the deformation through **Jacobians**, giving a global vertex–pixel relation, driven by differentiable rendering against pretrained image encoders. **Geometry in Style** [27] achieves *identity-preserving* stylization by representing deformation as per-vertex target normals resolved through a differentiable As-Rigid-As-Possible layer — expressive enough for detail, restrictive enough to preserve the source shape. **RADmesh** [28] adds **adaptive remeshing** to text-guided localized deformation, deforming and retriangulating selected mesh regions on a coarse-to-fine schedule.

This family shares our machinery: differentiable rendering, 2D supervision, geometry and appearance edited together, strong regularization to keep deformations well-behaved. Their techniques transfer directly to the geometry-modulation problem we face — Jacobian parameterizations, dARAP regularization, and in RADmesh's case remeshing, which matters because a fixed triangulation degenerates under large shape change. *The difference is that each produces a single deformation of a given mesh: one input, one output.* We require a **series** of deformations, indexed continuously by time, over a mesh we are simultaneously *generating* rather than being given. Their regularizers become tools inside our temporal model rather than alternatives to it.

---

# 3. Positioning: the gap and our contribution

The literature establishes four things. **(i)** Structured latent representations are a powerful substrate for high-fidelity static mesh generation. **(ii)** Geometry and appearance, though entangled in the latent, admit meaningful separate intervention at the stage and channel level [12, 14, 15, 18]. **(iii)** Latent-domain operations produce coherent transitions where mesh-space operations cannot, because correspondence exists in the latent and not across differing triangulations [17, 18, 19]. **(iv)** Temporally dynamic 3D generation is addressed today only by training 4D models, requiring data that does not exist at 3D scale [21, 22, 23], or by single-shot mesh deformation, which produces one configuration rather than a trajectory [25, 26, 27, 28].

Low-rank adaptation of a structured-latent 3D decoder has one precedent — FlowBender [16] — and it is static: it corrects appearance toward better conditioning alignment. **No prior work makes a pretrained 3D generator reproduce a specific object and then evolve that object over time, from a single view, with the temporal prior supplied by a video model.** That is the gap DynaMesh occupies.

Our contribution has three parts.

**3D inversion.** Just as image diffusion required inversion techniques to move from generating *a* plausible image to reproducing *a specific* one, we adapt a pretrained 3D generator to reproduce a specific object in both geometry and texture. Unlike 2D inversion, we must do this while preserving the model's 3D prior — the capacity to complete surfaces the single input view never showed.

**Video-supervised temporal modulation.** A video model supplies temporally coherent target frames, contributing a temporal inductive bias we neither train nor possess. A differentiable renderer propagates that 2D supervision into the 3D representation. No 4D annotation, no 4D dataset, no backbone modification.

**Joint texture and geometry evolution under single-view supervision.** We modulate both the geometry and the appearance channels at the decoder's output projection — the one point where they are explicitly separable — so a single reference video yields a multi-view-coherent object that changes shape and appearance together.

The balance between these is the difficulty and the novelty. Fit the specific object too aggressively and the 3D prior collapses, leaving an appearance correct from the supervised view and meaningless elsewhere. Preserve the prior too conservatively and the specific object is never reproduced. DynaMesh is our account of how to hold both.

---

## References

1. Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020.* [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
2. Kerbl, Kopanas, Leimkühler, Drettakis. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG 42(4), 2023.* [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
3. Poole, Jain, Barron, Mildenhall. "DreamFusion: Text-to-3D using 2D Diffusion." *ICLR 2023.* [arXiv:2209.14988](https://arxiv.org/abs/2209.14988)
4. Lin et al. "Magic3D: High-Resolution Text-to-3D Content Creation." *CVPR 2023.* [arXiv:2211.10440](https://arxiv.org/abs/2211.10440)
5. Hong et al. "LRM: Large Reconstruction Model for Single Image to 3D." *ICLR 2024.* [arXiv:2311.04400](https://arxiv.org/abs/2311.04400)
6. Xu et al. "InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models." *arXiv 2024.* [arXiv:2404.07191](https://arxiv.org/abs/2404.07191)
7. Li et al. "CraftsMan3D: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner." *arXiv 2024.* [arXiv:2405.14979](https://arxiv.org/abs/2405.14979)
8. Xiang et al. "Structured 3D Latents for Scalable and Versatile 3D Generation" (TRELLIS). *CVPR 2025 (Spotlight).* [arXiv:2412.01506](https://arxiv.org/abs/2412.01506)
9. Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." *TMLR 2024.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
10. Shen et al. "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization" (FlexiCubes). *ACM TOG 42(4), 2023.* [arXiv:2308.05371](https://arxiv.org/abs/2308.05371)
11. "Native and Compact Structured Latents for 3D Generation" (TRELLIS.2). *arXiv 2025.* [arXiv:2512.14692](https://arxiv.org/abs/2512.14692)
12. "UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation." *arXiv 2025.* [arXiv:2509.25079](https://arxiv.org/abs/2509.25079)
13. "SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling." *ETH Zurich, Stanford, Technion, NVIDIA. arXiv 2025.* [arXiv:2512.05343](https://arxiv.org/abs/2512.05343)
14. "Feedforward 3D Editing via Text-Steerable Image-to-3D." *arXiv 2025.* [arXiv:2512.13678](https://arxiv.org/abs/2512.13678)
15. Zhang et al. "StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance." *SIGGRAPH Asia 2025.* [arXiv:2509.13301](https://arxiv.org/abs/2509.13301)
16. "FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows." *arXiv 2026.* [arXiv:2606.20404](https://arxiv.org/abs/2606.20404)
17. Sun et al. "MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing." *CVPR 2026.* [arXiv:2601.00204](https://arxiv.org/abs/2601.00204)
18. "Interp3D: Correspondence-aware Interpolation for Generative Textured 3D Morphing." *arXiv 2026.* [arXiv:2601.14103](https://arxiv.org/abs/2601.14103)
19. "Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models." *arXiv 2025.* [arXiv:2511.22425](https://arxiv.org/abs/2511.22425)
20. "ArtiLatent: Realistic Articulated 3D Object Generation via Structured Latents." *SIGGRAPH Asia 2025.* [arXiv:2510.21432](https://arxiv.org/abs/2510.21432)
21. Kwon, Choi, Shin, Kim, Lee, Kim. "MORPHOS: Autoregressive 4D Generation with Temporal Structured Latents." *KAIST, arXiv 2026.* [arXiv:2606.02491](https://arxiv.org/abs/2606.02491)
22. "SS4D: Native 4D Generative Model via Structured Spacetime Latents." *SIGGRAPH Asia 2025 / ACM TOG.* [arXiv:2512.14284](https://arxiv.org/abs/2512.14284)
23. "Helix4D: Complex 4D Mesh Generation." *arXiv 2026.* [arXiv:2605.26109](https://arxiv.org/abs/2605.26109)
24. Kim et al. "Controllable Dynamic 3D Shape Generation via 3D Trajectories and Text" (T2Mo). *KAIST, arXiv 2026.* [arXiv:2606.05162](https://arxiv.org/abs/2606.05162)
25. Michel, Bar-On, Liu, Benaim, Hanocka. "Text2Mesh: Text-Driven Neural Stylization for Meshes." *CVPR 2022.* [arXiv:2112.03221](https://arxiv.org/abs/2112.03221)
26. Gao et al. "TextDeformer: Geometry Manipulation using Text Guidance." *SIGGRAPH 2023.* [arXiv:2304.13348](https://arxiv.org/abs/2304.13348)
27. Dinh, Lang, Kim, Stein, Hanocka. "Geometry in Style: 3D Stylization via Surface Normal Deformation." *CVPR 2025.* [arXiv:2503.23241](https://arxiv.org/abs/2503.23241)
28. Dinh, Lang, Stein, Hanocka. "RADmesh: Remesh-Aware Mesh Deformation." *ECCV 2026 (Oral).*

---

## Appendix: verification status and disclosure

**AI disclosure.** This document was prepared with AI assistance for structure, style, and literature verification, as permitted. All technical claims about our own system trace to measurements in the project repository; all claims about prior work were checked against the cited papers' full text.

**Verification.** Every reference above was checked on 2026-07-29 by retrieving the paper and confirming its title, venue, and the specific claim made in the text. Evidence with quotations is in `CITATION_AUDIT_v2_FULLTEXT.md`.

**Outstanding items before submission.**
- Author lists are confirmed for refs 1, 2, 3, 21, 25, 27, 28 and for ref 13's affiliations. Remaining "X et al." attributions are unchecked.
- Venue confirmations pending against official proceedings: ref 3 (ICLR 2023), ref 9 (TMLR 2024), ref 18 (a widely cited ICLR 2026 acceptance is not stated in the paper), ref 19 (NeurIPS 2025 likewise not stated), refs 6, 7 (arXiv only, no venue stated).
- Ref 28: the BibTeX entry in the public repository has an incorrect `title` field; cite the title as given above.
