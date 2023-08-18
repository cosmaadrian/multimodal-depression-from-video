#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=perceiver-window-size

# NOTA BENE: n_temporal_windows = 1, behaves like the baseline (i.e. no temporal windows)

python main.py --run_id 1 --name window-size-10-run-1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name window-size-10-run-2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name window-size-10-run-3 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name window-size-5-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name window-size-5-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name window-size-5-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name window-size-2-run-1 --seconds_per_window 2 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name window-size-2-run-2 --seconds_per_window 2 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name window-size-2-run-3 --seconds_per_window 2 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name window-size-1-run-1 --seconds_per_window 1 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name window-size-1-run-2 --seconds_per_window 1 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name window-size-1-run-3 --seconds_per_window 1 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings


