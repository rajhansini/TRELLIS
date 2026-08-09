#!/bin/bash
# Rung 5 Sub-ladder submit script.
#
# Stages:
#   5.0   baseline eval (no GPU time limit needed, fast)
#   5.1   output head only (AppearanceHeadLoRA, 576 params)
#   5.2   last block only (block 11, all 4 layers, 49,152 params)
#   5.3   placement sweep: early/mid/late thirds (3 parallel jobs)
#   5.4   all 12 blocks (589,824 params)
#   5.5   rank sweep on winner (3 parallel jobs: r4, r8, r16)
#
# Usage:
#   bash submit_rung5.sh [5.0|5.1|5.2|5.3|5.4|5.5] [--rank R] [--epochs E]
#   bash submit_rung5.sh 5.3              # launch all three placements
#   bash submit_rung5.sh 5.5 --all-ranks  # sweep r4, r8, r16 in parallel
#
# REPLACED: this file replaces the old prototype rung5_decoder_lora submit.
# New script: rung5_subladder.py (--stage flag covers all stages above)

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung5_subladder.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

# ── parse args ─────────────────────────────────────────────────────────────────
TARGET="${1:-help}"
shift || true

RANK=4
EPOCHS=30
SEED=6
ACTIVE_BLOCKS_55="8,9,10,11"   # default winner for 5.5 (dec-late)
ALL_RANKS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)          RANK="$2";             shift 2 ;;
        --epochs)        EPOCHS="$2";           shift 2 ;;
        --seed)          SEED="$2";             shift 2 ;;
        --active-blocks) ACTIVE_BLOCKS_55="$2"; shift 2 ;;
        --all-ranks)     ALL_RANKS=1;           shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$TARGET" == "help" ]] || [[ "$TARGET" == "" ]]; then
    echo "Usage: bash submit_rung5.sh [5.0|5.1|5.2|5.3|5.4|5.5] [--rank R] [--epochs E]"
    exit 0
fi

echo "=== Rung 5 Sub-ladder ==="
echo "  target=$TARGET  rank=$RANK  epochs=$EPOCHS  seed=$SEED"
echo ""

# ── shared build job ──────────────────────────────────────────────────────────
_submit_build() {
    sbatch \
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
" | awk '{print $NF}'
}

# ── generic training job submit ───────────────────────────────────────────────
# $1=stage  $2=dep_jobid (empty=no dep)  extra args follow
_submit_stage() {
    local STAGE="$1"
    local DEP="$2"
    shift 2
    local EXTRA="$*"
    local DEP_FLAG=""
    [[ -n "$DEP" ]] && DEP_FLAG="--dependency=afterok:${DEP}"

    sbatch \
        --job-name="r5_${STAGE//./_}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        --requeue \
        ${DEP_FLAG} \
        --output="${LOGDIR}/rung5_${STAGE//./_}_%j.log" \
        --error="${LOGDIR}/rung5_${STAGE//./_}_%j.log" \
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
echo '=== Rung 5: stage=${STAGE} rank=${RANK} seed=${SEED} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} -u ${SCRIPT} --stage ${STAGE} --rank ${RANK} --seed ${SEED} --epochs ${EPOCHS} ${EXTRA}
echo '=== Done: ${STAGE} ==='
" | awk '{print $NF}'
}

# ── 5.0: baseline eval ────────────────────────────────────────────────────────
if [[ "$TARGET" == "5.0" ]]; then
    echo "=== Submitting 5.0 (baseline eval) ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    J50=$(_submit_stage "5.0" "$BUILD_JOB")
    echo "  5.0 eval: $J50"
    echo ""
fi

# ── 5.1: output head only ─────────────────────────────────────────────────────
if [[ "$TARGET" == "5.1" ]]; then
    echo "=== Submitting 5.1 (output head LoRA) ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    J51=$(_submit_stage "5.1" "$BUILD_JOB")
    echo "  5.1 head: $J51"
    echo ""
fi

# ── 5.2: last block ───────────────────────────────────────────────────────────
if [[ "$TARGET" == "5.2" ]]; then
    echo "=== Submitting 5.2 (last block) ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    J52=$(_submit_stage "5.2" "$BUILD_JOB")
    echo "  5.2 last-block: $J52"
    echo ""
fi

# ── 5.3: placement sweep (3 parallel jobs) ────────────────────────────────────
if [[ "$TARGET" == "5.3" ]]; then
    echo "=== Submitting 5.3 placement sweep (early / mid / late) ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    J53E=$(_submit_stage "5.3-early" "$BUILD_JOB")
    J53M=$(_submit_stage "5.3-mid"   "$BUILD_JOB")
    J53L=$(_submit_stage "5.3-late"  "$BUILD_JOB")
    echo "  5.3-early: $J53E"
    echo "  5.3-mid:   $J53M"
    echo "  5.3-late:  $J53L"
    echo ""
    # Cross-placement gate check (waits for all three to confirm identical init renders)
    GATE3_JOB=$(sbatch \
        --job-name="r5_gate3_53" \
        --partition=general \
        --cpus-per-task=2 \
        --mem=4G \
        --time=00:10:00 \
        --dependency=afterok:${J53E}:${J53M}:${J53L} \
        --output="${LOGDIR}/rung5_53_gate3_%j.log" \
        --error="${LOGDIR}/rung5_53_gate3_%j.log" \
        --wrap="
set -e
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
echo '=== Gate 3-cross: 5.3 placement init render comparison ==='

\$PY - <<'EOF'
import torch, sys
from pathlib import Path

gate3 = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung5_gate3')
files = {p.stem.replace('init_', ''): p for p in gate3.glob('init_5_3_*.pt')}
print(f'Found: {sorted(files.keys())}')
if len(files) < 3:
    print('ERROR: fewer than 3 5.3 init renders — check that all three runs saved gate output')
    sys.exit(1)

tensors = {k: torch.load(v, weights_only=True) for k,v in files.items()}
TOL = 0.05
ok = True
keys = sorted(tensors.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        d = (tensors[a] - tensors[b]).abs().max().item()
        s = 'PASS' if d < TOL else 'FAIL'
        print(f'  {a} vs {b}: max_diff={d:.3e}  (tol={TOL})  [{s}]')
        if d >= TOL: ok = False
print()
if ok:
    print('GATE 3-cross PASSED — all 5.3 placements share the same init render.')
    print('Placement sweep is valid: structure, noise, and decoder are all fixed.')
else:
    print('GATE 3-cross FAILED — init renders differ. Check --seed and N_vox match.')
    sys.exit(1)
EOF
echo '=== Gate 3-cross done ==='
" | awk '{print $NF}')
    echo "  gate3-cross: $GATE3_JOB  (depends on $J53E:$J53M:$J53L)"
    echo ""
fi

# ── 5.4: all blocks ───────────────────────────────────────────────────────────
if [[ "$TARGET" == "5.4" ]]; then
    echo "=== Submitting 5.4 (all 12 blocks) ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    J54=$(_submit_stage "5.4" "$BUILD_JOB")
    echo "  5.4 all-blocks: $J54"
    echo ""
fi

# ── 5.5: rank sweep on winner ─────────────────────────────────────────────────
if [[ "$TARGET" == "5.5" ]]; then
    echo "=== Submitting 5.5 rank sweep  active-blocks=${ACTIVE_BLOCKS_55} ==="
    BUILD_JOB=$(_submit_build)
    echo "  build: $BUILD_JOB"
    if [[ "$ALL_RANKS" == "1" ]]; then
        for R in 4 8 16; do
            J=$(_submit_stage "5.5" "$BUILD_JOB" "--rank $R --active-blocks ${ACTIVE_BLOCKS_55}")
            echo "  5.5 r${R}: $J"
        done
    else
        J=$(_submit_stage "5.5" "$BUILD_JOB" "--active-blocks ${ACTIVE_BLOCKS_55}")
        echo "  5.5 r${RANK}: $J"
    fi
    echo ""
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo "=== All jobs submitted ==="
echo ""
echo "Monitor:"
echo "  squeue -u rajhansini"
echo ""
echo "Tail a log:"
echo "  ls ${LOGDIR}/rung5_*.log | sort | tail -3 | while read f; do echo \"=== \$f ===\"; tail -5 \"\$f\"; done"
echo ""
echo "Results after completion:"
echo "  ls experiments/lora_experiments/runs/ | grep rung5"
