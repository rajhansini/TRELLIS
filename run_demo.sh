#!/usr/bin/env bash
# =============================================================================
#  TRELLIS — Dynamic Texture Modulation Demo Script
#  Branch: trellis_modulation_coefficient
#
#  Usage:
#    bash run_demo.sh [--skip-tryrun] [--train-steps N] [--data-dir PATH]
#
#  Stages:
#    1. Environment sanity check
#    2. Smoke test  (no real data needed, ~30 s)
#    3. Create dummy VideoConditionedSLat sequences (if real data absent)
#    4. --tryrun   (model instantiation + summary, ~2 min)
#    5. Short real training (optional, only if --train-steps given)
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_TRYRUN=0
TRAIN_STEPS=0
DATA_DIR=""
CONDA_ENV="trellis"
CONDA_ENV_PREFIX="/net/projects/ranalab/rajhansini/conda_envs/trellis"
OUTPUT_DIR="outputs/dynamic_texture_demo"

# Use xformers backend if flash_attn is unavailable on this node (GLIBC issue)
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
CONFIG="configs/generation/dynamic_texture_flow_img_dit_L_64l8p2_fp16.json"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tryrun)  SKIP_TRYRUN=1 ; shift ;;
    --train-steps)  TRAIN_STEPS="$2" ; shift 2 ;;
    --data-dir)     DATA_DIR="$2" ; shift 2 ;;
    *) echo "Unknown arg: $1" ; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${GREEN}══ $* ${NC}"; }
warn() { echo -e "${YELLOW}WARN: $*${NC}"; }
die()  { echo -e "${RED}FATAL: $*${NC}"; exit 1; }

# ── cd to repo root ───────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Stage 0: Environment
# =============================================================================
step "0/5  Environment check"

# Detect and activate conda
_CONDA_ACTIVATED=0
if ! command -v conda &>/dev/null; then
  for _prefix in /opt/conda "${HOME}/miniconda3" "${HOME}/anaconda3"; do
    if [[ -f "${_prefix}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${_prefix}/etc/profile.d/conda.sh" && _CONDA_ACTIVATED=1 && break
    fi
  done
fi

if [[ "${_CONDA_ACTIVATED}" -eq 0 ]] && command -v conda &>/dev/null; then
  _CONDA_ACTIVATED=1
fi
[[ "${_CONDA_ACTIVATED}" -eq 1 ]] || die "conda not found — activate it manually first"

# Activate the trellis env by prefix (non-standard install path)
CURRENT_ENV="${CONDA_DEFAULT_ENV:-base}"
if [[ "${CURRENT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "  Activating conda env prefix: ${CONDA_ENV_PREFIX}"
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "${CONDA_ENV_PREFIX}" 2>/dev/null || \
    conda activate "${CONDA_ENV}" 2>/dev/null || \
    die "Failed to activate conda env '${CONDA_ENV}' or '${CONDA_ENV_PREFIX}'"
fi
echo "  Python: $(python --version)"
echo "  Active env: ${CONDA_DEFAULT_ENV:-${CONDA_ENV_PREFIX}}"
echo "  ATTN_BACKEND: ${ATTN_BACKEND}"

# Check CUDA
python -c "
import torch
avail = torch.cuda.is_available()
print(f'  PyTorch {torch.__version__}, CUDA available: {avail}')
if not avail:
    import sys
    print('  WARNING: No GPU detected. spconv requires CUDA — forward-pass stages will be skipped.')
    print('  Run this script on a GPU node (f002/f003) for full verification.')
"

# Check we're on the right branch
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "  Git branch: ${BRANCH}"
[[ "${BRANCH}" == "trellis_modulation_coefficient" ]] || \
  warn "Expected branch 'trellis_modulation_coefficient', got '${BRANCH}'"

echo -e "  ${GREEN}Environment OK${NC}"

# =============================================================================
# Stage 1: Smoke test
# =============================================================================
step "1/5  Smoke test — temporal modulation forward pass"
SMOKE_LOG="${LOG_DIR}/smoke_test_${TIMESTAMP}.log"
echo "  Log: ${SMOKE_LOG}"

python scripts/smoke_test.py 2>&1 | tee "${SMOKE_LOG}"
SMOKE_EXIT="${PIPESTATUS[0]}"
if [[ "${SMOKE_EXIT}" -ne 0 ]]; then
  die "Smoke test FAILED — see ${SMOKE_LOG}"
fi
echo -e "  ${GREEN}Smoke test PASSED${NC}"

# =============================================================================
# Stage 2: Data setup
# =============================================================================
step "2/5  Data setup"

DYNAMIC_SEQ_ROOT="data/dynamic_sequences"

if [[ -n "${DATA_DIR}" ]]; then
  echo "  Using provided data dir: ${DATA_DIR}"
  DYNAMIC_SEQ_ROOT="${DATA_DIR}"
elif [[ -d "${DYNAMIC_SEQ_ROOT}" ]] && \
     python -c "
import os, json, sys
root = '${DYNAMIC_SEQ_ROOT}'
seqs = [d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root,d))
        and os.path.exists(os.path.join(root,d,'metadata.json'))]
print(f'  Found {len(seqs)} sequence(s) in {root}')
sys.exit(0 if seqs else 1)
" 2>/dev/null; then
  echo -e "  ${GREEN}Using existing sequences in ${DYNAMIC_SEQ_ROOT}${NC}"
else
  echo "  No real sequences found — creating dummy data for tryrun ..."
  python scripts/create_dummy_sequences.py --root "${DYNAMIC_SEQ_ROOT}" \
    2>&1 | tee "${LOG_DIR}/create_dummy_${TIMESTAMP}.log"
  echo -e "  ${GREEN}Dummy data created${NC}"
fi

# =============================================================================
# Stage 3: tryrun (model instantiation + param summary)
# =============================================================================
if [[ "${SKIP_TRYRUN}" -eq 1 ]]; then
  warn "Skipping --tryrun (--skip-tryrun flag set)"
else
  step "3/5  --tryrun — instantiate model + print architecture"
  TRYRUN_LOG="${LOG_DIR}/tryrun_${TIMESTAMP}.log"
  echo "  Output dir: ${OUTPUT_DIR}"
  echo "  Log: ${TRYRUN_LOG}"

  python train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus 1 \
    --tryrun \
    2>&1 | tee "${TRYRUN_LOG}"

  TRYRUN_EXIT="${PIPESTATUS[0]}"
  if [[ "${TRYRUN_EXIT}" -ne 0 ]]; then
    die "--tryrun FAILED — see ${TRYRUN_LOG}"
  fi
  echo -e "  ${GREEN}--tryrun PASSED${NC}"
  echo "  Model summary written to: ${OUTPUT_DIR}/denoiser_model_summary.txt"
fi

# =============================================================================
# Stage 4: Optional short training run
# =============================================================================
if [[ "${TRAIN_STEPS}" -gt 0 ]]; then
  step "4/5  Short training run (${TRAIN_STEPS} steps)"
  TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
  echo "  Log: ${TRAIN_LOG}"

  # Patch config on the fly: reduce batch size and steps for the demo
  DEMO_CONFIG="${OUTPUT_DIR}/demo_config.json"
  python - <<PYEOF
import json, copy, sys

with open("${CONFIG}") as f:
    cfg = json.load(f)

# Shrink for quick demo
cfg["trainer"]["args"]["max_steps"]         = ${TRAIN_STEPS}
cfg["trainer"]["args"]["batch_size_per_gpu"] = 1
cfg["trainer"]["args"]["batch_split"]        = 1
cfg["trainer"]["args"]["i_log"]             = 1
cfg["trainer"]["args"]["i_save"]            = 99999
cfg["trainer"]["args"]["i_sample"]          = 99999

with open("${DEMO_CONFIG}", "w") as f:
    json.dump(cfg, f, indent=4)
print("Demo config written to: ${DEMO_CONFIG}")
PYEOF

  python train.py \
    --config "${DEMO_CONFIG}" \
    --output_dir "${OUTPUT_DIR}/train_run" \
    --num_gpus 1 \
    2>&1 | tee "${TRAIN_LOG}"

  TRAIN_EXIT="${PIPESTATUS[0]}"
  if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
    die "Training FAILED — see ${TRAIN_LOG}"
  fi
  echo -e "  ${GREEN}Training run PASSED${NC}"
else
  step "4/5  Training run — SKIPPED (pass --train-steps N to enable)"
fi

# =============================================================================
# Summary
# =============================================================================
step "5/5  Summary"
echo ""
echo "  Branch:      ${BRANCH}"
echo "  Smoke test:  ${SMOKE_LOG}"
[[ "${SKIP_TRYRUN}" -eq 0 ]] && echo "  Tryrun:      ${TRYRUN_LOG}" || true
[[ "${TRAIN_STEPS}" -gt 0  ]] && echo "  Train log:   ${TRAIN_LOG}"  || true
echo ""
echo "  Key files changed on this branch:"
git diff main --name-only 2>/dev/null | sed 's/^/    /'
echo ""
echo -e "${GREEN}All stages completed successfully.${NC}"
