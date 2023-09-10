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

A_MODALITIES="daic_audio_covarep daic_audio_formant"
V_MODALITIES="daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"
AV_MODALITIES="daic_audio_covarep daic_audio_formant daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"

WANDB_MODE=run
GROUP=perceiver-daicwoz-ablation-modalities

TRAINING_ARGS='--trainer '$TRAINER' --scheduler '$SCHEDULER' --epochs '$EPOCHS' --batch_size '$BATCH_SIZE' --scheduler_args.max_lr '$MAX_LR' --scheduler_args.end_epoch '$EPOCHS
MODEL_ARGS='--model_args.latent_num 16 --model_args.latent_dim 128 --model_args.context_dim 256 --model_args.cross_attn_num_heads 4 --model_args.cross_attn_dim_head 64 --model_args.self_attn_num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64'

# # AUDIO
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audio-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 2 --save_model 2 --group $GROUP --mode $WANDB_MODE --name audio-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 3 --save_model 3 --group $GROUP --mode $WANDB_MODE --name audio-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 4 --save_model 4 --group $GROUP --mode $WANDB_MODE --name audio-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $A_MODALITIES
python main.py --run_id 5 --save_model 5 --group $GROUP --mode $WANDB_MODE --name audio-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $A_MODALITIES

# # VIDEO
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name video-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 2 --save_model 2 --group $GROUP --mode $WANDB_MODE --name video-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 3 --save_model 3 --group $GROUP --mode $WANDB_MODE --name video-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 4 --save_model 4 --group $GROUP --mode $WANDB_MODE --name video-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $V_MODALITIES
python main.py --run_id 5 --save_model 5 --group $GROUP --mode $WANDB_MODE --name video-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $V_MODALITIES

# # AUDIOVISUAL
python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 2 --save_model 2 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 3 --save_model 3 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 4 --save_model 4 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AV_MODALITIES
python main.py --run_id 5 --save_model 5 --group $GROUP --mode $WANDB_MODE --name audiovisual-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AV_MODALITIES

