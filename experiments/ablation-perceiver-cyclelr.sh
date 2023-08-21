#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=perceiver-cyclelr


for LR in 0.00001 0.0001 0.001 
do
    for RUN_ID in 1 2 3
        do
        WANDB_NAME=cyclelr-lr-$LR-run-$RUN_ID
        python main.py --run_id $RUN_ID --name $WANDB_NAME --trainer temporal --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $LR --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
    done
done