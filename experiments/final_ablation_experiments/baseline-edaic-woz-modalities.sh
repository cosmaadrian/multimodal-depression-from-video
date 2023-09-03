#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=250
SECONDS_PER_WINDOW=6

GROUP=baseline-e-daic-woz-modalities

A_MODALITIES="edaic_audio_mfcc edaic_audio_egemaps"
V_MODALITIES="edaic_video_cnn_resnet edaic_video_pose_gaze_aus"
AV_MODALITIES="edaic_audio_mfcc edaic_audio_egemaps edaic_video_cnn_resnet edaic_video_pose_gaze_aus"

python main.py --run_id 1 --name audio-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 2 --name audio-run-2 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 3 --name audio-run-3 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $A_MODALITIES

python main.py --run_id 1 --name video-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 2 --name video-run-2 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 3 --name video-run-3 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $V_MODALITIES

python main.py --run_id 1 --name audiovisual-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 2 --name audiovisual-run-2 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 3 --name audiovisual-run-3 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/edaicwoz_config.yaml --dataset e-daic-woz --env banamar-upv16 --use_modalities $AV_MODALITIES
