#!/bin/bash

set -e
cd ..

EPOCHS=200
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.0001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.50
SECONDS_PER_WINDOW=9

AV='orig_face_landmarks orig_audio_descriptors'

MODEL_ARGS="--model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32"

WANDB_MODE=dryrun
ENV="reading-between-the-frames"
CONFIG_FILE="configs/train_configs/baseline_original_dvlog_new_split_config.yaml"
GROUP=baseline-original-dvlog-new-split-modality-ablation

# AV
python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --dataset original-d-vlog-new-split --use_modalities $AV
python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --dataset original-d-vlog-new-split --use_modalities $AV
python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --dataset original-d-vlog-new-split --use_modalities $AV
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --dataset original-d-vlog-new-split --use_modalities $AV
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --dataset original-d-vlog-new-split --use_modalities $AV
