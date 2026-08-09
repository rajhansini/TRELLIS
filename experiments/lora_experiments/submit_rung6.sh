#!/bin/bash
# Rung 6 — DINOv2 Encoder LoRA submit script.
#
# Two-step submit:
#   BUILD_JOB : rebuild nvdiffrast for the cluster GPU
#   TRAIN_JOB : rung6_dino_lora.py, depends on build
#
# IMPORTANT: Rung 6 is the slowest rung. It cannot pre-compute DINOv2 tokens
# because the LoRA is active during encoding. Each epoch re-encodes all 135
# training frames through DINOv2 (24 blocks) and propagates gradients through
# the LONGEST path of all rungs:
#   DINOv2+LoRA -> cond_gl -> last denoising step -> slat -> decode -> render
#
# Only run Rung 6 if Rungs 1-5 leave an unexplained quality gap (spec Part B).
#
# Usage:
#   bash submit_rung6.sh                        # defaults: rank=4 layers=attn blocks=all epochs=20
#   bash submit_rung6.sh --rank 8               # pass any rung6 flag
#   bash submit_rung6.sh --layers all --rank 4  # include fc1/fc2 MLP layers
#   bash submit_rung6.sh --smoke                # 2-epoch dev smoke test
#
# Logs:
#   Slurm build : slurm_logs/build_a40_<jobid>.log
#   Slurm train : slurm_logs/rung6_train_<jobid>.log
#   Python      : runs/rung6_dino_*/train.log   (tee'd inside rung6_dino_lora.py)

set -e

PY=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/rung6_dino_lora.py
BUILD_SCRIPT=/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/build_renderers_A40.sh
LOGDIR=/net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/slurm_logs
mkdir -p "$LOGDIR"

# All CLI arguments are forwarded to rung6_dino_lora.py
RUNG6_ARGS="$@"

echo "=== Rung 6 — DINOv2 Encoder LoRA ==="
echo "  args: ${RUNG6_ARGS:-<defaults>}"
echo ""
echo "  NOTE: This is the slowest rung (no token pre-computation)."
echo "        Wall-time budget: 8h for 20 epochs."
echo ""

# ── Step 1: rebuild nvdiffrast for A40 (sm_86) ────────────────────────────────
echo "=== Step 1: nvdiffrast build ==="

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
echo "  Build log : ${LOGDIR}/build_a40_${BUILD_JOB}.log"
echo ""

# ── Step 2: Rung 6 DINOv2 encoder LoRA training ──────────────────────────────
# Gradient injection path (see rung6_dino_lora.py docstring):
#   1. encode_with_lora (grad on LoRA params)
#   2. 24 prefix denoising steps: no_grad (cond_gl used, graph not tracked)
#   3. last denoising step: grad ON (cond_gl enters computation graph)
#   4. decode + render (slat_leaf detaches from flow)
#   5. loss.backward() -> slat_leaf.grad
#   6. torch.autograd.backward(slat.feats, slat_leaf.grad)
#      -> propagates through last step -> cond_gl -> DINOv2 LoRA A, B
# Time budget: 8h for 20 epochs (conservative for re-encode overhead).
echo "=== Step 2: Rung 6 training (depends on build ${BUILD_JOB}) ==="

TRAIN_JOB=$(sbatch \
    --job-name="rung6_dino_lora" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=08:00:00 \
    --requeue \
    --dependency=afterok:${BUILD_JOB} \
    --output="${LOGDIR}/rung6_train_%j.log" \
    --error="${LOGDIR}/rung6_train_%j.log" \
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
echo '=== Rung 6: DINOv2 encoder LoRA ==='
cd /net/projects/ranalab/rajhansini/TRELLIS

${PY} ${SCRIPT} ${RUNG6_ARGS}

echo '=== Rung 6 complete ==='
" | awk '{print $NF}')

echo "  Train job : ${TRAIN_JOB}  (waits on build ${BUILD_JOB})"
echo "  Slurm log : ${LOGDIR}/rung6_train_${TRAIN_JOB}.log"
echo "  Python log: /net/projects/ranalab/rajhansini/TRELLIS/experiments/lora_experiments/runs/rung6_dino_*/train.log"
echo ""
echo "=== Submitted (build ${BUILD_JOB} → train ${TRAIN_JOB}) ==="
echo ""
echo "Monitor:    squeue -u rajhansini"
echo "Tail build: tail -f ${LOGDIR}/build_a40_${BUILD_JOB}.log"
echo "Tail train: tail -f ${LOGDIR}/rung6_train_${TRAIN_JOB}.log"
echo ""
echo "Key output files (once the hash is known):"
echo "  experiments/lora_experiments/runs/rung6_dino_attn_all_r4_s6_*/loss_history.json"
echo "  experiments/lora_experiments/runs/rung6_dino_attn_all_r4_s6_*/final_eval.json"
echo "  experiments/lora_experiments/runs/rung6_dino_attn_all_r4_s6_*/diag_renders/"
echo ""
echo "If B.grad stays near zero after epoch 1, try: --lr 1e-3"
