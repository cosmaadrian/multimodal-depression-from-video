#!/bin/bash

set -e
cd ..

EPOCHS=250
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.50
SECONDS_PER_WINDOW=9

LANDMARKS='face_landmarks body_landmarks hand_landmarks'
EYES='gaze_features blinking_features'
AUDIO_VISUAL='audio_embeddings face_embeddings'
ALL_MODALITIES='audio_embeddings face_embeddings face_landmarks body_landmarks hand_landmarks gaze_features blinking_features'

MODEL_ARGS="--model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32"

WANDB_MODE=run
ENV="banamar-upv16"
GROUP=dvlog-baseline-ablation-modalities-final


# Audio-visual + landmarks + eyes
# python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-eyes-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $ALL_MODALITIES
# python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-eyes-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $ALL_MODALITIES
# python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-eyes-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $ALL_MODALITIES

# Audio-visual + landmarks
# python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-spw-9-pt-0.50-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-spw-9-pt-0.50-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS
# python main.py --run_id 6 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-lm-spw-9-pt-0.50-run-6 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS

# # # Audio-visual + eyes
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-eyes-spw-9-pt-0.50-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-eyes-spw-9-pt-0.50-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS
python main.py --run_id 6 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-eyes-spw-9-pt-0.50-run-6 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $AUDIO_VISUAL $LANDMARKS

# # Landmarks + eyes
# python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-eyes-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS $EYES
# python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-eyes-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS $EYES
# python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-eyes-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS $EYES

# # Landmarks (face, hands, body)
# python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS
# python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS
# python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name lm-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $LANDMARKS

# # Eyes (blinking + gaze)
# python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name eyes-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $EYES
# python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name eyes-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $EYES
# python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name eyes-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env $ENV $MODEL_ARGS --use_modalities $EYES
