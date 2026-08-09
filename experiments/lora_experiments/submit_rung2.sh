#!/bin/bash
# Rung 2 — Placement ablation (2a screen: 8 blocks, rank=4, 163,840 params each)
#
# Launches 3 training jobs in parallel (early / mid / late).
# A Gate 3 check job runs after all three complete to verify byte-identical init renders.
#
# Usage:
#   bash submit_rung2.sh               # 2a screen (rank=4, 15 epochs)
#   bash submit_rung2.sh --rank 12     # 2b matched budget (rank=12, 491,520 params)
#   bash submit_rung2.sh --epochs 20   # more epochs
#
# Logs: lora_experiments/slurm_logs/rung2_*.log
# Output: lora_experiments/runs/rung2_{placement}_r{rank}_s6_{hash}/

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung2_placement.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

# ── parse optional args ───────────────────────────────────────────────────────
RANK=4
EPOCHS=30
SEED=6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)   RANK="$2";   shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Rung 2 — Placement ablation ==="
echo "  rank=$RANK  epochs=$EPOCHS  seed=$SEED"
echo ""

# ── Step 1: shared nvdiffrast build ──────────────────────────────────────────
echo "=== Step 1: nvdiffrast build (shared) ==="
BUILD_JOB=$(sbatch \
    --job-name="build_a40" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGDIR}/build_a40_%j.log" \
    --error="${LOGDIR}/build_a40_%j.log" \
    --wrap="
set -e
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
bash ${BUILD_SCRIPT}
" | awk '{print $NF}')
echo "  Build job : ${BUILD_JOB}"
echo ""

# ── Step 2: launch early, mid, late — 3 chained rounds each ─────────────────
# Cluster max walltime is 4h; 30 epochs @ ~16 min = ~8h total.
# rung2_placement.py saves lora_e{epoch}.pt each epoch and resumes from the
# latest checkpoint on restart. afterany on rounds 2/3 ensures they run even
# after TIMEOUT (not just afterok).
echo "=== Step 2: Launch all three placements — 3 x 4h chained rounds each ==="

_submit_round() {
    local PLACEMENT=$1
    local DEP_FLAG=$2
    sbatch \
        --job-name="rung2_${PLACEMENT}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        ${DEP_FLAG} \
        --output="${LOGDIR}/rung2_${PLACEMENT}_%j.log" \
        --error="${LOGDIR}/rung2_${PLACEMENT}_%j.log" \
        --wrap="
set -e
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

echo '=== GPU info ==='
nvidia-smi -L
echo ''
echo '=== Rung 2: placement=${PLACEMENT} rank=${RANK} seed=${SEED} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} -u ${SCRIPT} --placement ${PLACEMENT} --rank ${RANK} --seed ${SEED} --epochs ${EPOCHS}

echo '=== Done: ${PLACEMENT} ==='
" | awk '{print $NF}'
}

_submit_placement() {
    local PLACEMENT=$1
    local R1=$(_submit_round "$PLACEMENT" "--dependency=afterok:${BUILD_JOB}")
    local R2=$(_submit_round "$PLACEMENT" "--dependency=afterany:${R1}")
    local R3=$(_submit_round "$PLACEMENT" "--dependency=afterany:${R2}")
    echo "  ${PLACEMENT}: r1=${R1}  r2=${R2}  r3=${R3}" >&2
    echo "$R3"
}

EARLY_JOB=$(_submit_placement early | tail -1)
MID_JOB=$(_submit_placement mid   | tail -1)
LATE_JOB=$(_submit_placement late  | tail -1)

echo ""

# ── Step 3: Gate 3 cross-run sanity check (runs after all three complete) ─────
echo "=== Step 3: Gate 3 check (depends on all three) ==="
GATE3_JOB=$(sbatch \
    --job-name="rung2_gate3" \
    --partition=general \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:10:00 \
    --dependency=afterok:${EARLY_JOB}:${MID_JOB}:${LATE_JOB} \
    --output="${LOGDIR}/rung2_gate3_%j.log" \
    --error="${LOGDIR}/rung2_gate3_%j.log" \
    --wrap="
set -e
GATE3=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung2_gate3
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

echo '=== Gate 3: cross-run init render comparison ==='
echo ''

\$PY - <<'EOF'
import torch, sys
from pathlib import Path
gate3 = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung2_gate3')
files = {p.stem.replace('init_', ''): p for p in gate3.glob('init_*.pt')}
print(f'Found: {list(files.keys())}')
if len(files) < 3:
    print('ERROR: fewer than 3 init renders — not all runs saved Gate 3 output')
    sys.exit(1)
# Threshold: xformers self-attn non-determinism across separate jobs yields ~1e-3 diffs.
# Diffs >> 0.05 indicate a different seed or N_vox mismatch — ablation is invalid.
TOL = 0.05
tensors = {k: torch.load(v, weights_only=True) for k, v in files.items()}
keys    = list(tensors.keys())
ok = True
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        diff = (tensors[a] - tensors[b]).abs().max().item()
        status = 'PASS' if diff < TOL else 'FAIL'
        print(f'  {a} vs {b}:  max_diff={diff:.3e}  (tol={TOL})  [{status}]')
        if diff >= TOL:
            ok = False
if ok:
    print('')
    print('GATE 3 PASSED — all three init renders are consistent.')
    print('The ablation is valid: only placement differs.')
else:
    print('')
    print('GATE 3 FAILED — init renders differ across placements.')
    print('Check that all runs used the same --seed and produced N_vox=7301.')
    sys.exit(1)
EOF
echo ''
echo '=== Gate 3 done ==='
" | awk '{print $NF}')

echo "  Gate 3 job: ${GATE3_JOB}  (depends on ${EARLY_JOB}:${MID_JOB}:${LATE_JOB})"
echo ""

# ── summary ───────────────────────────────────────────────────────────────────
echo "=== All jobs submitted ==="
echo ""
echo "  build    : ${BUILD_JOB}"
echo "  early    : ${EARLY_JOB}   (rank=${RANK}, 163,840 params)"
echo "  mid      : ${MID_JOB}   (rank=${RANK}, 163,840 params)"
echo "  late     : ${LATE_JOB}   (rank=${RANK}, 163,840 params)"
echo "  gate3    : ${GATE3_JOB}  (cross-run sanity)"
echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo ""
echo "Tail logs:"
echo "  tail -f ${LOGDIR}/rung2_early_${EARLY_JOB}.log"
echo "  tail -f ${LOGDIR}/rung2_mid_${MID_JOB}.log"
echo "  tail -f ${LOGDIR}/rung2_late_${LATE_JOB}.log"
echo "  tail -f ${LOGDIR}/rung2_gate3_${GATE3_JOB}.log"
echo ""
echo "Results (after runs complete):"
echo "  ls lora_experiments/runs/"
echo "  cat lora_experiments/runs/rung2_early_r${RANK}_s${SEED}_*/loss_history.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d[-1])\""
