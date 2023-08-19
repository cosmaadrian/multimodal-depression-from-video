#!/bin/bash

set -e
cd ..

WANDB_MODE=dryrun
BATCH_SIZE=8
EPOCHS=500
GROUP=perceiver-n-windows

python main.py --run_id 1 --name n-windows-1-run-1 --presence_threshold -1 --trainer temporal --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name n-windows-1-run-2 --presence_threshold -1 --trainer temporal --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name n-windows-1-run-3 --presence_threshold -1 --trainer temporal --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name n-windows-2-run-1 --presence_threshold -1 --trainer temporal --n_temporal_windows 2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name n-windows-2-run-2 --presence_threshold -1 --trainer temporal --n_temporal_windows 2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name n-windows-2-run-3 --presence_threshold -1 --trainer temporal --n_temporal_windows 2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name n-windows-4-run-1 --presence_threshold -1 --trainer temporal --n_temporal_windows 4 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name n-windows-4-run-2 --presence_threshold -1 --trainer temporal --n_temporal_windows 4 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name n-windows-4-run-3 --presence_threshold -1 --trainer temporal --n_temporal_windows 4 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name n-windows-6-run-1 --presence_threshold -1 --trainer temporal --n_temporal_windows 6 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name n-windows-6-run-2 --presence_threshold -1 --trainer temporal --n_temporal_windows 6 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name n-windows-6-run-3 --presence_threshold -1 --trainer temporal --n_temporal_windows 6 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --name n-windows-8-run-1 --presence_threshold -1 --trainer temporal --n_temporal_windows 8 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --name n-windows-8-run-2 --presence_threshold -1 --trainer temporal --n_temporal_windows 8 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --name n-windows-8-run-3 --presence_threshold -1 --trainer temporal --n_temporal_windows 8 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings