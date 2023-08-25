#!/bin/bash

set -e
cd ..

WANDB_MODE=dryrun
BATCH_SIZE=8
EPOCHS=500
GROUP=baseline-d-vlog-original

python main.py --run_id 1 --name baseline-run-1 --save_model 1 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --config_file configs/original_dvlog_config.yaml --dataset original-d-vlog --env bucuram-exodus --use_modalities orig_face_landmarks, orig_audio_descriptors