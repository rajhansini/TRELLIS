#!/bin/bash
# Watches a SLURM training job and resubmits when it dies until epoch 50 is done.
# Usage: nohup bash auto_resume.sh <job_id> > ../results/auto_resume.log 2>&1 &

SUBMIT_SCRIPT="/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/step8_train/submit_step8_train.sh"
CKPT_DIR="/net/projects/ranalab/rajhansini/TRELLIS/experiments/dynamic_texture_trellis_pipeline/results/lora_ckpts"
FINAL_CKPT="$CKPT_DIR/lora_e050.pt"

JOB_ID=$1
if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

echo "[$(date)] auto_resume started, watching job $JOB_ID"

while true; do
    # Poll until job leaves the queue
    while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 60
    done

    echo "[$(date)] Job $JOB_ID is no longer in queue"

    # Done if epoch 50 checkpoint exists
    if [ -f "$FINAL_CKPT" ]; then
        echo "[$(date)] Training complete — $FINAL_CKPT found. Exiting."
        exit 0
    fi

    LATEST=$(ls "$CKPT_DIR"/lora_e*.pt 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST" ]; then
        echo "[$(date)] Latest checkpoint: $(basename "$LATEST"). Resubmitting..."
    else
        echo "[$(date)] No checkpoint found. Resubmitting from scratch..."
    fi

    NEW_JOB=$(sbatch "$SUBMIT_SCRIPT" | grep -oP '\d+')
    if [ -z "$NEW_JOB" ]; then
        echo "[$(date)] ERROR: sbatch failed. Retrying in 60s..."
        sleep 60
        continue
    fi

    echo "[$(date)] Submitted new job $NEW_JOB"
    JOB_ID=$NEW_JOB
done
