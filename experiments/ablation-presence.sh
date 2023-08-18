#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=500
GROUP=baseline-presence-ablation-2

python main.py --run_id 1 --presence_threshold 0.1 --name presence-0.1-window-10-run-1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.1 --name presence-0.1-window-10-run-2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.1 --name presence-0.1-window-10-run-3 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --presence_threshold 0.01 --name presence-0.01-window-10-run-1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.01 --name presence-0.01-window-10-run-2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.01 --name presence-0.01-window-10-run-3 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --presence_threshold 0.5 --name presence-0.5-window-10-run-1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.5 --name presence-0.5-window-10-run-2 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.5 --name presence-0.5-window-10-run-3 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

################################################
################################################
################################################
################################################
python main.py --run_id 1 --presence_threshold 0.1 --name presence-0.1-window-5-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.1 --name presence-0.1-window-5-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.1 --name presence-0.1-window-5-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --presence_threshold 0.01 --name presence-0.01-window-5-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.01 --name presence-0.01-window-5-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.01 --name presence-0.01-window-5-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings

python main.py --run_id 1 --presence_threshold 0.5 --name presence-0.5-window-5-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 2 --presence_threshold 0.5 --name presence-0.5-window-5-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings
python main.py --run_id 3 --presence_threshold 0.5 --name presence-0.5-window-5-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings