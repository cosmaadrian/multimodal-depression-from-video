#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=baseline-cosinelr

# python main.py --run_id 1 --name cosinelr-0.00001-run-1 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.00001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
# python main.py --run_id 2 --name cosinelr-0.00001-run-2 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.00001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
# python main.py --run_id 3 --name cosinelr-0.00001-run-3 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.00001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

# python main.py --run_id 1 --name cosinelr-0.0001-run-1 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.0001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name cosinelr-0.0001-run-2 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.0001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name cosinelr-0.0001-run-3 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.0001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name cosinelr-0.001-run-1 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name cosinelr-0.001-run-2 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name cosinelr-0.001-run-3 --save_model 0 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
