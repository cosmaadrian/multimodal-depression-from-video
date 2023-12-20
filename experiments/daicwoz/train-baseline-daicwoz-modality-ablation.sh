#!/bin/bash

set -e
cd ..

EPOCHS=200
MAX_LR=0.0001
BATCH_SIZE=8
SECONDS_PER_WINDOW=6

A="daic_audio_covarep daic_audio_formant"
V="daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"
AV="daic_audio_covarep daic_audio_formant daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"

WANDB_MODE=dryrun
MODEL_ARGS="--model_args.num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64"
GROUP=baseline-daicwoz-modality-ablation
CONFIG_FILE="configs/train_configs/baseline_daicwoz_config.yaml"
ENV="reading-between-the-frames"

# A
python main.py --run_id 1 --name audio-run-1 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $A
python main.py --run_id 2 --name audio-run-2 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $A
python main.py --run_id 3 --name audio-run-3 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $A
python main.py --run_id 4 --name audio-run-4 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $A
python main.py --run_id 5 --name audio-run-5 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $A

# V
python main.py --run_id 1 --name video-run-1 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $V
python main.py --run_id 2 --name video-run-2 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $V
python main.py --run_id 3 --name video-run-3 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $V
python main.py --run_id 4 --name video-run-4 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $V
python main.py --run_id 5 --name video-run-5 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $V

# AV
python main.py --run_id 1 --name audiovisual-run-1 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 2 --name audiovisual-run-2 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 3 --name audiovisual-run-3 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 4 --name audiovisual-run-4 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 5 --name audiovisual-run-5 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset daic-woz --env $ENV --use_modalities $AV
