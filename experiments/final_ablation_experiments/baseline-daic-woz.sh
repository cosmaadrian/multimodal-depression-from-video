#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=250
GROUP=baseline-daic-woz
USE_MODALITIES="daic_audio_covarep daic_audio_formant daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"

python main.py --run_id 1 --name baseline-run-1 --save_model 1 --n_temporal_windows 1 --seconds_per_window 6 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/daicwoz_config.yaml --dataset daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 2 --name baseline-run-2 --save_model 1 --n_temporal_windows 1 --seconds_per_window 6 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/daicwoz_config.yaml --dataset daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 3 --name baseline-run-3 --save_model 1 --n_temporal_windows 1 --seconds_per_window 6 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/daicwoz_config.yaml --dataset daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
