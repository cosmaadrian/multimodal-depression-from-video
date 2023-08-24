#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=baseline-linearlr

python main.py --run_id 1 --name lr-0.001-1 --scheduler_args.max_lr 0.001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name lr-0.001-2 --scheduler_args.max_lr 0.001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name lr-0.001-3 --scheduler_args.max_lr 0.001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name lr-0.0001-1 --scheduler_args.max_lr 0.0001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name lr-0.0001-2 --scheduler_args.max_lr 0.0001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name lr-0.0001-3 --scheduler_args.max_lr 0.0001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name lr-0.00001-1 --scheduler_args.max_lr 0.00001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name lr-0.00001-2 --scheduler_args.max_lr 0.00001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name lr-0.00001-3 --scheduler_args.max_lr 0.00001 --scheduler linear --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
