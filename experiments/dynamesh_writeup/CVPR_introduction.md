# DynaMesh: Video-Supervised Appearance Adaptation of Generated 3D Meshes

*CVPR submission — Abstract and Introduction*

---

## Abstract

We present **DynaMesh**, a framework that adapts a frozen 3D generative model to reproduce the evolving appearance of a specific object, supervised by a single video sequence. A lightweight low-rank adapter is anchored at the output projection layer of a frozen TRELLIS decoder — the first and only point in the pipeline where geometry and appearance channels become explicitly separable, and the only position at which a per-voxel edit provably cannot propagate to other voxels. A pretrained video model supplies temporally coherent target frames; a differentiable renderer propagates that 2D supervision into the 3D representation. We frame the problem as **3D inversion**: adapting a generative model built to produce *plausible* objects so that it reproduces a *specific* one, while preserving the 3D prior that lets it complete surfaces the single input view never showed. The generative backbone is never modified and no 4D supervision is used. Geometry is held fixed by construction, which makes the intervention a controlled edit of appearance alone.

---

## 1. Introduction

Generative AI has changed how 3D content is made. Modern systems synthesize expressive, detailed 3D shapes from a sentence or a photograph in seconds, and that capability is being absorbed into augmented and virtual reality, film and game production, product visualization, and simulation — because it removes the bottleneck that mattered most: getting a plausible, usable 3D object into the scene at all.

Yet almost everything these systems produce is frozen at the moment of creation. The geometry is fixed, the texture is baked onto its surface, and the asset cannot subsequently change. For much of the physical world this is the wrong model. Real objects age, dry, corrode, bruise, and wilt, and their appearance shifts as they do. A generative model that captures the *what* of an object but has no account of the *when* cannot express any of this.

**Making a generated 3D asset change over time has scarcely been explored**, and the two established routes both struggle with it. Native 4D generators [21, 22, 23] require large-scale 4D supervision that does not exist at the scale 3D data does — SS4D, for instance, fine-tunes both the autoencoder and generator of its 3D backbone and curates 16,000 animated objects to do so. Mesh deformation under a differentiable-rendering objective [25, 26, 27, 28] yields a *single* new configuration rather than a progression through configurations.

We introduce **DynaMesh**, which adapts a frozen 3D generative model to reproduce a specific object's changing appearance, supervised by a single video of the target effect.

**Two pretrained models, one mechanism.** Our approach rests on a division of labour. A 3D generative model contributes spatial understanding: from one view it infers a complete object, including surfaces the camera never saw. A video generative model contributes temporal understanding: an inductive bias for smooth, physically plausible change over time. Neither can do the other's job, and our contribution is the mechanism that couples them.

**The problem is inversion, not generation.** A 3D generative model is built to produce *plausible* objects, not *particular* ones. Asked for a teapot it returns a good teapot — rarely *this* teapot. The situation is directly analogous to 2D image diffusion, where generating a specific image rather than a plausible one required a distinct body of inversion techniques. We therefore frame our problem as **3D inversion**: adapt a pretrained 3D generator so that it reproduces a specific object faithfully, while retaining the 3D prior that completes unobserved regions. Unlike 2D inversion, fidelity and prior preservation are in direct tension: overfit the observed view and the object becomes a decal, correct from one angle and meaningless elsewhere.

**Where we intervene, and why it is the only viable position.** TRELLIS [8] generates in two stages — a sparse structure stage that decides where the object is, and a latent stage that assigns each occupied voxel a compact code encoding its local surface detail and appearance *jointly*. Because that same code is consumed by mesh, Gaussian, and radiance-field decoders, geometry and appearance are not separable at the latent. They separate exactly once, at the decoder's output projection, where each fine voxel's features are mapped to explicitly partitioned geometry and appearance channels.

That layer is also the only place a per-voxel edit stays local. Every earlier stage mixes information across voxels — windowed self-attention in the transformer blocks, and convolution in the upsamplers — so an adapter placed upstream propagates an edit intended for one region into others. We show this empirically: adapters placed inside the attention blocks, and adapters placed on the block MLPs with attention untouched, both corrupt regions we never targeted. The output projection is a per-voxel linear map with nothing downstream that mixes, so an edit there provably cannot spread. **Identifying this constraint, and the position that satisfies it, is one of our contributions.**

**Supervision.** A video model provides temporally coherent target frames; a differentiable renderer carries that 2D signal into the 3D representation. Supervision comes from a **single viewpoint**, and we hold the sparse structure and initial latent noise fixed across the sequence so that frame-to-frame variation is driven by the video conditioning rather than by sampling noise. Geometry channels are left untouched, making this a controlled intervention on appearance whose effect on shape is exactly zero by construction rather than by regularization.

**Contributions.**
- **3D inversion of a pretrained generator.** We adapt a frozen 3D generative model to reproduce a specific object's appearance while preserving its capacity to complete unseen surfaces, using a small number of trainable parameters and no modification to the backbone.
- **Identification of the sole contamination-free intervention point.** We characterize where in a structured-latent decoder a per-voxel edit remains local, and show empirically that the two natural alternatives — inside attention, and on the block MLPs — both leak.
- **Video-supervised appearance adaptation without 4D data.** A pretrained video model supplies the temporal signal at supervision time, so no 4D dataset, no 4D annotation, and no new temporal weights are required. The one prior instance of low-rank adaptation in this family, FlowBender [16], corrects appearance toward better conditioning alignment on a static asset; we use the same paradigm to track a video sequence.

**Scope.** We modulate appearance channels only. Extending the same mechanism to the geometry channels — so that shape and appearance change together — is the natural next step and is discussed in Section 6. We treat the backbone as an implementation detail: we build on TRELLIS, but the framework assumes only a generator with a per-element latent and a decoder in which geometry and appearance are separable, properties shared by an emerging family of structured-latent 3D models.

---

## References cited in this section

8. Xiang et al. "Structured 3D Latents for Scalable and Versatile 3D Generation" (TRELLIS). *CVPR 2025 (Spotlight).* [arXiv:2412.01506](https://arxiv.org/abs/2412.01506)
16. "FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows." *arXiv 2026.* [arXiv:2606.20404](https://arxiv.org/abs/2606.20404)
21. Kwon et al. "MORPHOS: Autoregressive 4D Generation with Temporal Structured Latents." *arXiv 2026.* [arXiv:2606.02491](https://arxiv.org/abs/2606.02491)
22. "SS4D: Native 4D Generative Model via Structured Spacetime Latents." *SIGGRAPH Asia 2025 / ACM TOG.* [arXiv:2512.14284](https://arxiv.org/abs/2512.14284)
23. "Helix4D: Complex 4D Mesh Generation." *arXiv 2026.* [arXiv:2605.26109](https://arxiv.org/abs/2605.26109)
25. Michel, Bar-On, Liu, Benaim, Hanocka. "Text2Mesh: Text-Driven Neural Stylization for Meshes." *CVPR 2022.* [arXiv:2112.03221](https://arxiv.org/abs/2112.03221)
26. Gao et al. "TextDeformer: Geometry Manipulation using Text Guidance." *SIGGRAPH 2023.* [arXiv:2304.13348](https://arxiv.org/abs/2304.13348)
27. Dinh, Lang, Kim, Stein, Hanocka. "Geometry in Style: 3D Stylization via Surface Normal Deformation." *CVPR 2025.* [arXiv:2503.23241](https://arxiv.org/abs/2503.23241)
28. Dinh, Lang, Stein, Hanocka. "RADmesh: Remesh-Aware Mesh Deformation." *ECCV 2026 (Oral).*

*Numbering matches `GRANT_intro_related_work.md` so the two share one bibliography.*
