#!/bin/bash
# Rung 5 full pipeline — runs unattended: 5.1 → 5.2 → 5.3(E/M/L) → 5.4 → 5.5
#
# Each stage uses 3 × 4h chained rounds (afterany) for TIMEOUT resilience.
# rung5_subladder.py saves a checkpoint each epoch and resumes on restart.
# Stages advance only when the final round of the previous stage succeeds (afterok).
#
# Usage:
#   bash submit_rung5_pipeline.sh [--rank 4] [--epochs 30] [--from 5.1]
#
# --from 5.2  restarts the pipeline from stage 5.2 with no prior dependency.
# --from 5.3  starts from 5.3, etc.

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung5_subladder.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

RANK=4
EPOCHS=30
SEED=6
FROM_STAGE="5.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)   RANK="$2";       shift 2 ;;
        --epochs) EPOCHS="$2";     shift 2 ;;
        --seed)   SEED="$2";       shift 2 ;;
        --from)   FROM_STAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Rung 5 Full Pipeline ==="
echo "  stages: 5.1 → 5.2 → 5.3(E/M/L) → 5.4 → 5.5"
echo "  rank=$RANK  epochs=$EPOCHS  seed=$SEED  from=${FROM_STAGE}"
echo ""

# ── helpers ────────────────────────────────────────────────────────────────────

# Submit a build job. $1 = optional afterok job IDs (colon-separated, or "").
_submit_build() {
    local PREV_IDS="$1"
    local DEP_FLAG=""
    [[ -n "$PREV_IDS" ]] && DEP_FLAG="--dependency=afterok:${PREV_IDS}"
    sbatch \
        --job-name="build_a40" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=16G \
        --time=00:30:00 \
        ${DEP_FLAG} \
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

# Submit one 4h training slot. Returns job ID on stdout; prints info to stderr.
# $1=stage $2=dep_flag_string (full "--dependency=..." or "") $3+=extra python args
_submit_slot() {
    local STAGE="$1"
    local DEP_FLAG="$2"
    shift 2
    local EXTRA="$*"
    sbatch \
        --job-name="r5_${STAGE//./_}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
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
echo '=== Rung5 stage=${STAGE} rank=${RANK} seed=${SEED} epochs=${EPOCHS} ==='
cd /net/projects/ranalab/rajhansini/TRELLIS
${PY} -u ${SCRIPT} --stage ${STAGE} --rank ${RANK} --seed ${SEED} --epochs ${EPOCHS} ${EXTRA}
echo '=== Done: ${STAGE} ==='
" | awk '{print $NF}'
}

# Submit 3 chained rounds for a stage. Prints final round job ID to stdout.
# $1=stage  $2=build_job_id  $3=prev_stage_ids (colon-sep, or "")  $4+=extra python args
_submit_stage_chain() {
    local STAGE="$1"
    local BUILD_JOB="$2"
    local PREV_IDS="$3"
    shift 3
    local EXTRA="$*"

    # Round 1 waits for build + previous stage final round(s)
    local R1_DEP="afterok:${BUILD_JOB}"
    [[ -n "$PREV_IDS" ]] && R1_DEP="${R1_DEP}:${PREV_IDS}"
    local R1=$(_submit_slot "$STAGE" "--dependency=${R1_DEP}" "$EXTRA")

    # Rounds 2/3 use afterany so they run even after TIMEOUT
    local R2=$(_submit_slot "$STAGE" "--dependency=afterany:${R1}" "$EXTRA")
    local R3=$(_submit_slot "$STAGE" "--dependency=afterany:${R2}" "$EXTRA")

    echo "  ${STAGE}: r1=${R1}  r2=${R2}  r3=${R3}" >&2
    echo "$R3"
}

# ── stage 5.1 ──────────────────────────────────────────────────────────────────

PREV_IDS=""   # colon-separated list of final-round job IDs from previous stage

if [[ "$FROM_STAGE" == "5.1" ]]; then
    echo "--- Stage 5.1: output head LoRA (576 params) ---"
    BUILD_51=$(_submit_build "$PREV_IDS")
    echo "  build: $BUILD_51"
    J51=$(_submit_stage_chain "5.1" "$BUILD_51" "$PREV_IDS")
    echo "  5.1 final round: $J51"
    PREV_IDS="$J51"
    echo ""
fi

# ── stage 5.2 ──────────────────────────────────────────────────────────────────

if [[ "$FROM_STAGE" == "5.1" || "$FROM_STAGE" == "5.2" ]]; then
    echo "--- Stage 5.2: last decoder block (block 11, ~49k params) ---"
    BUILD_52=$(_submit_build "$PREV_IDS")
    echo "  build: $BUILD_52"
    J52=$(_submit_stage_chain "5.2" "$BUILD_52" "$PREV_IDS")
    echo "  5.2 final round: $J52"
    PREV_IDS="$J52"
    echo ""
fi

# ── stage 5.3: placement sweep ─────────────────────────────────────────────────

if [[ "$FROM_STAGE" == "5.1" || "$FROM_STAGE" == "5.2" || "$FROM_STAGE" == "5.3" ]]; then
    echo "--- Stage 5.3: decoder placement sweep (early/mid/late) ---"
    BUILD_53=$(_submit_build "$PREV_IDS")
    echo "  build: $BUILD_53"
    J53E=$(_submit_stage_chain "5.3-early" "$BUILD_53" "$PREV_IDS")
    J53M=$(_submit_stage_chain "5.3-mid"   "$BUILD_53" "$PREV_IDS")
    J53L=$(_submit_stage_chain "5.3-late"  "$BUILD_53" "$PREV_IDS")
    echo "  5.3 final rounds: early=$J53E  mid=$J53M  late=$J53L"
    PREV_IDS="${J53E}:${J53M}:${J53L}"
    echo ""
fi

# ── stage 5.4 ──────────────────────────────────────────────────────────────────

if [[ "$FROM_STAGE" == "5.1" || "$FROM_STAGE" == "5.2" || \
      "$FROM_STAGE" == "5.3" || "$FROM_STAGE" == "5.4" ]]; then
    echo "--- Stage 5.4: all 12 decoder blocks (~590k params) ---"
    BUILD_54=$(_submit_build "$PREV_IDS")
    echo "  build: $BUILD_54"
    J54=$(_submit_stage_chain "5.4" "$BUILD_54" "$PREV_IDS")
    echo "  5.4 final round: $J54"
    PREV_IDS="$J54"
    echo ""
fi

# ── stage 5.5: rank sweep on dec-late ──────────────────────────────────────────

echo "--- Stage 5.5: rank sweep (r4/r8/r16) on dec-late (blocks 8-11) ---"
BUILD_55=$(_submit_build "$PREV_IDS")
echo "  build: $BUILD_55"
J55R4=$( _submit_stage_chain "5.5" "$BUILD_55" "$PREV_IDS" "--rank 4  --active-blocks 8,9,10,11")
J55R8=$( _submit_stage_chain "5.5" "$BUILD_55" "$PREV_IDS" "--rank 8  --active-blocks 8,9,10,11")
J55R16=$(_submit_stage_chain "5.5" "$BUILD_55" "$PREV_IDS" "--rank 16 --active-blocks 8,9,10,11")
echo "  5.5 final rounds: r4=$J55R4  r8=$J55R8  r16=$J55R16"
echo ""

echo "=== Full rung5 pipeline submitted — runs unattended ==="
echo ""
echo "Monitor:"
echo "  squeue -u rajhansini | grep -v bash"
echo ""
echo "Tail latest log:"
echo "  ls ${LOGDIR}/rung5_*.log | sort | tail -1 | xargs tail -f"
