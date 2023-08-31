#!/bin/bash

set -e
cd ../

MIN_WINDOW_SIZE=$1
MAX_WINDOW_SIZE=$2

EPOCHS=250
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.25
USE_MODALITIES="audio_embeddings face_embeddings"

WANDB_MODE=run
ENV="banamar-upv16"
GROUP=baseline-ws-class-weights-ablation-final
DEFAULT_ARGS="--group $GROUP --mode $WANDB_MODE --env $ENV --config_file configs/baseline_config.yaml --save_model 1 --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --presence_threshold $PRESENCE_THRESHOLD --epochs $EPOCHS --batch_size $BATCH_SIZE --use_modalities $USE_MODALITIES"


for SECONDS_PER_WINDOW in $(seq $MIN_WINDOW_SIZE $MAX_WINDOW_SIZE)
do
    for RUN_ID in 1 2 3
        do
        WANDB_NAME="window-size-$SECONDS_PER_WINDOW-run-$RUN_ID"
        python main.py --run_id $RUN_ID --name $WANDB_NAME $DEFAULT_ARGS --seconds_per_window $SECONDS_PER_WINDOW
    done
done
