#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=250
SECONDS_PER_WINDOW=6

GROUP=baseline-e-daic-woz-model-size-ablation
USE_MODALITIES="edaic_audio_mfcc edaic_audio_egemaps edaic_video_cnn_resnet edaic_video_pose_gaze_aus"

# python main.py --run_id 1 --name nl-8-nh-8-dh-32-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 2 --name nl-8-nh-8-dh-32-run-2 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 3 --name nl-8-nh-8-dh-32-run-3 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES

python main.py --run_id 1 --name nl-8-nh-4-dh-64-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 2 --name nl-8-nh-4-dh-64-run-2 --model_args.num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
python main.py --run_id 3 --name nl-8-nh-4-dh-64-run-3 --model_args.num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $USE_MODALITIES
