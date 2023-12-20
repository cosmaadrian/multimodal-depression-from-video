#!/bin/bash

set -e
cd ..

EPOCHS=200
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.0001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
SECONDS_PER_WINDOW=6
PRESENCE_THRESHOLD=0.50

A="edaic_audio_mfcc edaic_audio_egemaps"
V="edaic_video_cnn_resnet edaic_video_pose_gaze_aus"
AV="edaic_audio_mfcc edaic_audio_egemaps edaic_video_cnn_resnet edaic_video_pose_gaze_aus"

WANDB_MODE=dryrun
GROUP=perceiver-edaic-modality-ablation

ENV="reading-between-the-frames"
TRAINING_ARGS='--trainer '$TRAINER' --scheduler '$SCHEDULER' --epochs '$EPOCHS' --batch_size '$BATCH_SIZE' --scheduler_args.max_lr '$MAX_LR' --scheduler_args.end_epoch '$EPOCHS
MODEL_ARGS='--model_args.latent_num 16 --model_args.latent_dim 128 --model_args.context_dim 256 --model_args.cross_attn_num_heads 8 --model_args.cross_attn_dim_head 32 --model_args.self_attn_num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32'
CONFIG_FILE="configs/train_configs/perceiver_edaic_config.yaml"

# A
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $A
python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $A
python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $A
python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $A
python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $A

# V
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $V
python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $V
python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $V
python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $V
python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $V

# AV
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV
python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV

