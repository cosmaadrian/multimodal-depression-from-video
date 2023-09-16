#!/bin/bash

set -e
cd ..

EPOCHS=200
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
SECONDS_PER_WINDOW=9
PRESENCE_THRESHOLD=0.50

ALL_MODALITIES='orig_audio_descriptors orig_face_landmarks'

WANDB_MODE=run
GROUP=perceiver-original-dvlog-new-split

TRAINING_ARGS='--trainer '$TRAINER' --scheduler '$SCHEDULER' --epochs '$EPOCHS' --batch_size '$BATCH_SIZE' --scheduler_args.max_lr '$MAX_LR' --scheduler_args.end_epoch '$EPOCHS
MODEL_ARGS='--model_args.latent_num 16 --model_args.latent_dim 128 --model_args.context_dim 256 --model_args.cross_attn_num_heads 8 --model_args.cross_attn_dim_head 32 --model_args.self_attn_num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32'

# # AV
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities $ALL_MODALITIES
