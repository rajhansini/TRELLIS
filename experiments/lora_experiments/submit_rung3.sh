#!/bin/bash
# Rung 3 — 2D Grid: bottom row (all-5 layers: to_q, to_kv, to_out, fc1, fc2)
#
# Launches 3 training jobs in parallel (early / mid / late).
# A Gate 3 check job runs after all three complete to verify consistent init renders.
#
# Usage:
#   bash submit_rung3.sh               # rank=4, 30 epochs
#   bash submit_rung3.sh --rank 4 --epochs 30
#
# Logs: lora_experiments/slurm_logs/rung3_*.log
# Output: lora_experiments/runs/rung3_{placement}_r{rank}_s{seed}_{hash}/
# Params: 557,056 per run (8 blocks × 17408 × rank=4)

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung3_grid.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

# ── parse optional args ────────────────────────────────────────────────────────
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

echo "=== Rung 3 — All-5 layers (bottom row of 2D grid) ==="
echo "  rank=$RANK  epochs=$EPOCHS  seed=$SEED"
echo "  params/run: $((8 * 17408 * RANK)) (8 blocks × 17408 × rank)"
echo ""

# ── Step 1: shared nvdiffrast build ───────────────────────────────────────────
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

# ── Step 2: launch early, mid, late — 3 chained rounds each ──────────────────
# Cluster max walltime is 4h; 30 epochs @ ~16 min = ~8h total.
# rung3_grid.py saves lora_e{epoch}.pt each epoch and resumes from the latest
# checkpoint on restart (same run dir because hash includes --epochs flag).
# Each round uses afterany so it runs even after TIMEOUT (not afterok).
echo "=== Step 2: Launch all three placements — 3 x 4h chained rounds each ==="

_submit_round() {
    local PLACEMENT=$1
    local DEP_FLAG=$2
    sbatch \
        --job-name="rung3_${PLACEMENT}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        ${DEP_FLAG} \
        --output="${LOGDIR}/rung3_${PLACEMENT}_%j.log" \
        --error="${LOGDIR}/rung3_${PLACEMENT}_%j.log" \
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
echo '=== Rung 3 all-5: placement=${PLACEMENT} rank=${RANK} seed=${SEED} ==='
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

# ── Step 3: Gate 3 cross-run sanity check ─────────────────────────────────────
echo "=== Step 3: Gate 3 check (depends on all three) ==="
GATE3_JOB=$(sbatch \
    --job-name="rung3_gate3" \
    --partition=general \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:10:00 \
    --dependency=afterok:${EARLY_JOB}:${MID_JOB}:${LATE_JOB} \
    --output="${LOGDIR}/rung3_gate3_%j.log" \
    --error="${LOGDIR}/rung3_gate3_%j.log" \
    --wrap="
set -e
GATE3_R3=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung3_gate3
GATE3_R2=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung2_gate3
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python

echo '=== Gate 3: rung3 bottom-row cross-run init render comparison ==='
echo ''

\$PY - <<'EOF'
import torch, sys
from pathlib import Path

gate3_r3 = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung3_gate3')
gate3_r2 = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung2_gate3')

# Bottom-row files (rung3)
files_r3 = {p.stem.replace('init_', ''): p for p in gate3_r3.glob('init_*.pt')}
print(f'Rung3 (bottom-row) files: {sorted(files_r3.keys())}')
if len(files_r3) < 3:
    print('ERROR: fewer than 3 rung3 init renders — not all runs saved G1-cross output')
    sys.exit(1)

# Top-row files (rung2) — for cross-row consistency
files_r2 = {p.stem.replace('init_', ''): p for p in gate3_r2.glob('init_*.pt')}
print(f'Rung2 (top-row) files:    {sorted(files_r2.keys())}')

# Threshold: xformers self-attn non-determinism across separate jobs yields ~1e-3 diffs.
# Diffs >> 0.05 indicate a different seed or N_vox mismatch.
TOL = 0.05
tensors_r3 = {k: torch.load(v, weights_only=True) for k, v in files_r3.items()}
tensors_r2 = {k: torch.load(v, weights_only=True) for k, v in files_r2.items()}

ok = True
print('')
print('-- Bottom-row (rung3) pairwise --')
keys = sorted(tensors_r3.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        diff = (tensors_r3[a] - tensors_r3[b]).abs().max().item()
        status = 'PASS' if diff < TOL else 'FAIL'
        print(f'  rung3:{a} vs rung3:{b}:  max_diff={diff:.3e}  (tol={TOL})  [{status}]')
        if diff >= TOL:
            ok = False

if tensors_r2:
    print('')
    print('-- Cross-row (rung3 vs rung2) --')
    for pl in sorted(tensors_r3.keys()):
        for pl2 in sorted(tensors_r2.keys()):
            diff = (tensors_r3[pl] - tensors_r2[pl2]).abs().max().item()
            status = 'PASS' if diff < TOL else 'FAIL'
            print(f'  rung3:{pl} vs rung2:{pl2}:  max_diff={diff:.3e}  (tol={TOL})  [{status}]')
            if diff >= TOL:
                ok = False

print('')
if ok:
    print('GATE 3 PASSED — all init renders are consistent across rows and placements.')
    print('The 2D grid ablation is valid: structure, noise, and conditioning are fixed.')
else:
    print('GATE 3 FAILED — init renders differ.')
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
echo "  early    : ${EARLY_JOB}   (all-5, rank=${RANK}, 557,056 params)"
echo "  mid      : ${MID_JOB}   (all-5, rank=${RANK}, 557,056 params)"
echo "  late     : ${LATE_JOB}   (all-5, rank=${RANK}, 557,056 params)"
echo "  gate3    : ${GATE3_JOB}  (cross-run + cross-row sanity)"
echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo ""
echo "Tail logs:"
echo "  tail -f ${LOGDIR}/rung3_early_${EARLY_JOB}.log"
echo "  tail -f ${LOGDIR}/rung3_mid_${MID_JOB}.log"
echo "  tail -f ${LOGDIR}/rung3_late_${LATE_JOB}.log"
echo "  tail -f ${LOGDIR}/rung3_gate3_${GATE3_JOB}.log"
echo ""
echo "Results (after runs complete):"
echo "  ls lora_experiments/runs/ | grep rung3"
echo "  cat lora_experiments/runs/rung3_early_r${RANK}_s${SEED}_*/loss_history.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d[-1])\""
