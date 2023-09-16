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

LANDMARKS='face_landmarks body_landmarks hand_landmarks'
EYES='gaze_features blinking_features'
AUDIO_VISUAL='audio_embeddings face_embeddings'
ALL_MODALITIES='audio_embeddings face_embeddings face_landmarks body_landmarks hand_landmarks gaze_features blinking_features'

WANDB_MODE=run
GROUP=perceiver-ablation-modalities

TRAINING_ARGS='--trainer '$TRAINER' --scheduler '$SCHEDULER' --epochs '$EPOCHS' --batch_size '$BATCH_SIZE' --scheduler_args.max_lr '$MAX_LR' --scheduler_args.end_epoch '$EPOCHS
MODEL_ARGS='--model_args.latent_num 16 --model_args.latent_dim 128 --model_args.context_dim 256 --model_args.cross_attn_num_heads 8 --model_args.cross_attn_dim_head 32 --model_args.self_attn_num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32'

# # AV + LM + EYES
# python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm+eyes-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
# python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm+eyes-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
# python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm+eyes-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
# python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm+eyes-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
# python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm+eyes-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES

# # AV + LM
# python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+lm-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS

# # AV + EYES
python main.py --run_id 6 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+eyes-run-6 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 7 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+eyes-run-7 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 8 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+eyes-run-8 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 9 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+eyes-run-9 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 10 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av+eyes-run-10 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES


# # AV
# python main.py --run_id 1 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-1 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL
# python main.py --run_id 2 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-2 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL
# python main.py --run_id 3 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-3 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL
# python main.py --run_id 4 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-4 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL
# python main.py --run_id 5 --save_model 1 --group $GROUP --mode $WANDB_MODE --name av-run-5 --presence_threshold $PRESENCE_THRESHOLD --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW  $TRAINING_ARGS $MODEL_ARGS --config_file configs/perceiver_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL
