#!/usr/bin/env bash
# Run the interpolation gap test on a GPU node.
# Usage: bash scripts/run_interp_gap_test.sh [--steps 6] [--no_gt]
#
# SLAT sampler steps:
#   12  = default quality   (~20 min on 1 GPU for 30 frames × GT + interp)
#    6  = 2x faster         (~10 min, still good quality)
#    3  = 4x faster         (~5  min, rough)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PREFIX="/net/projects/ranalab/rajhansini/conda_envs/trellis"

EXTRA_ARGS="${*:-}"   # pass through --steps N --no_gt etc.

export SPCONV_ALGO=native
export ATTN_BACKEND=xformers

cd "$REPO"

conda run -p "$CONDA_PREFIX" \
    python scripts/interpolation_gap_test.py $EXTRA_ARGS
