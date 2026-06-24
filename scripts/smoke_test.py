"""
Smoke test for the temporal modulation branch (trellis_modulation_coefficient).

Verifies WITHOUT any real data or checkpoints that:
  1. ElasticSLatFlowModel instantiates with use_temporal_modulation=True
  2. TemporalProjector has the expected architecture
  3. A forward pass with tau succeeds end-to-end
  4. Different tau values produce different outputs (once projector is non-trivial)

Run from the TRELLIS repo root:
    python scripts/smoke_test.py
"""

import sys
import os

# Default to xformers if ATTN_BACKEND not set (avoids GLIBC issues with flash_attn
# on nodes where the binary was compiled against a newer glibc).
os.environ.setdefault("ATTN_BACKEND", "xformers")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import trellis.modules.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model config (small for speed, mirrors the real L config but tiny) ────────
MODEL_CFG = dict(
    resolution=64,
    in_channels=8,
    out_channels=8,
    model_channels=64,       # 1024 in prod → 64 for smoke
    cond_channels=64,        # 1024 in prod → 64 for smoke
    num_blocks=2,            # 24 in prod → 2 for smoke
    num_heads=4,
    mlp_ratio=4,
    patch_size=2,
    num_io_res_blocks=2,
    io_block_channels=[32],
    pe_mode="ape",
    qk_rms_norm=True,
    use_fp16=False,          # fp32 on any GPU/CPU
    use_temporal_modulation=True,
    temporal_proj_hidden=64,
)

BATCH = 2
N_VOXELS = 512    # voxels per batch item
COND_TOKENS = 37  # DINOv2 patch tokens (37x37 → 1369, but use small N here)


def make_sparse(n_voxels, batch, device, channels):
    """Create a synthetic SparseTensor with random voxels."""
    coords_list = []
    feats_list = []
    for b in range(batch):
        coords = torch.randint(0, 32, (n_voxels, 3), dtype=torch.int32)
        coords = torch.cat([torch.full((n_voxels, 1), b, dtype=torch.int32), coords], dim=1)
        coords_list.append(coords)
        feats_list.append(torch.randn(n_voxels, channels))
    x = sp.SparseTensor(
        coords=torch.cat(coords_list).to(device),
        feats=torch.cat(feats_list).to(device),
    )
    return x


def main():
    print("=" * 60)
    print("TRELLIS Temporal Modulation — Smoke Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── 1. Import model ──────────────────────────────────────────────
    print("\n[1/5] Importing ElasticSLatFlowModel ...", end=" ", flush=True)
    from trellis.models.structured_latent_flow import ElasticSLatFlowModel, TemporalProjector
    print("OK")

    # ── 2. Instantiate ───────────────────────────────────────────────
    print("[2/5] Instantiating model with use_temporal_modulation=True ...", end=" ", flush=True)
    model = ElasticSLatFlowModel(**MODEL_CFG).to(DEVICE)
    print("OK")

    # ── 3. Verify projector architecture ────────────────────────────
    print("[3/5] Checking TemporalProjector ...")
    assert model.temporal_projector is not None, "temporal_projector is None!"
    tp = model.temporal_projector
    n_params = sum(p.numel() for p in tp.parameters())
    print(f"      TemporalProjector present: {type(tp).__name__}")
    print(f"      Parameters: {n_params:,}")
    print(f"      tau_embedder: {tp.tau_embedder}")
    print(f"      cond_proj:    {tp.cond_proj}")
    print("      OK")

    # ── 4. Forward pass with tau ─────────────────────────────────────
    if not torch.cuda.is_available():
        print("[4/5] Forward pass — SKIPPED (spconv requires CUDA, no GPU on this node)")
        print("[5/5] Tau sensitivity — SKIPPED (requires CUDA)")
        print("\n" + "=" * 60)
        print("ARCHITECTURE CHECKS PASSED (steps 1-3)")
        print("Re-run on a GPU node for the full forward-pass verification.")
        print("=" * 60)
        return

    print("[4/5] Running forward pass (batch=2, tau=[0.0, 1.0]) ...")
    model.eval()
    with torch.no_grad():
        x = make_sparse(N_VOXELS, BATCH, DEVICE, MODEL_CFG["in_channels"])
        t = torch.tensor([500.0, 500.0], device=DEVICE)
        cond = torch.randn(BATCH, COND_TOKENS, MODEL_CFG["cond_channels"], device=DEVICE)
        tau = torch.tensor([0.0, 1.0], device=DEVICE)

        out = model(x, t, cond, tau=tau)

    assert out.feats.shape == x.feats.shape, \
        f"Output shape mismatch: {out.feats.shape} vs {x.feats.shape}"
    assert torch.isfinite(out.feats).all(), "Non-finite values in output!"
    print(f"      Output shape: {out.feats.shape}  ✓")
    print("      All values finite  ✓")
    print("      OK")

    # ── 5. Verify tau sensitivity after non-trivial weights ──────────
    print("[5/5] Verifying tau sensitivity (non-zero projector weights) ...")
    # Re-init the final MLP layer with random weights so tau actually changes output.
    # (Production model starts zero-init for stable training.)
    with torch.no_grad():
        torch.nn.init.normal_(tp.mlp[-1].weight, std=0.1)
        torch.nn.init.normal_(tp.mlp[-1].bias, std=0.1)

    model.eval()
    with torch.no_grad():
        x0 = make_sparse(N_VOXELS, 1, DEVICE, MODEL_CFG["in_channels"])
        t1 = torch.tensor([500.0], device=DEVICE)
        cond1 = torch.randn(1, COND_TOKENS, MODEL_CFG["cond_channels"], device=DEVICE)

        out_early = model(x0, t1, cond1, tau=torch.tensor([0.0], device=DEVICE))
        out_late  = model(x0, t1, cond1, tau=torch.tensor([1.0], device=DEVICE))

    diff = (out_early.feats - out_late.feats).abs().mean().item()
    assert diff > 1e-6, f"tau=0.0 and tau=1.0 produced identical outputs (diff={diff:.2e})"
    print(f"      Mean |out(τ=0) - out(τ=1)| = {diff:.4f}  (tau sensitivity confirmed)  ✓")
    print("      OK")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — temporal modulation is working end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    main()
