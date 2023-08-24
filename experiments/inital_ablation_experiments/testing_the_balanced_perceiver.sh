#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=balanced-perceiver-cosinelr-n-windows

python main.py --run_id 1 --name cosinelr-0.00001-n-windows-1-run-1 --scheduler cosine --scheduler_args.max_lr 0.00001 --config_file configs/balanced_perceiver_config.yaml --save_model 1 --trainer temporal --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 1 --name cosinelr-0.00001-n-windows-2-run-1 --scheduler cosine --scheduler_args.max_lr 0.00001 --config_file configs/balanced_perceiver_config.yaml --save_model 1 --trainer temporal --n_temporal_windows 2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 1 --name cosinelr-0.00001-n-windows-4-run-1 --scheduler cosine --scheduler_args.max_lr 0.00001 --config_file configs/balanced_perceiver_config.yaml --save_model 1 --trainer temporal --n_temporal_windows 4 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 1 --name cosinelr-0.00001-n-windows-6-run-1 --scheduler cosine --scheduler_args.max_lr 0.00001 --config_file configs/balanced_perceiver_config.yaml --save_model 1 --trainer temporal --n_temporal_windows 6 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
