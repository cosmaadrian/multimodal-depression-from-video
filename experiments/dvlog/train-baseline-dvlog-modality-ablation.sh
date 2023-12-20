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

LM='face_landmarks body_landmarks hand_landmarks'
EYES='gaze_features blinking_features'
AV='audio_embeddings face_embeddings'

MODEL_ARGS="--model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32"

WANDB_MODE=dryrun
ENV="reading-between-the-frames"
CONFIG_FILE="configs/train_configs/baseline_dvlog_config.yaml"
GROUP=baseline-dvlog-modality-ablation

# AV
python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV
python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV
python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV

# AV + EYES
python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+eyes-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $EYES
python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+eyes-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $EYES
python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+eyes-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $EYES
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+eyes-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $EYES
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+eyes-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $EYES

# AV + LM
python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM
python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM
python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM

# AV + LM + EYES
python main.py --run_id 1 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm+eyes-run-1 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM $EYES
python main.py --run_id 2 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm+eyes-run-2 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM $EYES
python main.py --run_id 3 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm+eyes-run-3 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM $EYES
python main.py --run_id 4 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm+eyes-run-4 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM $EYES
python main.py --run_id 5 --save_model 1 --presence_threshold $PRESENCE_THRESHOLD --name av+lm+eyes-run-5 --seconds_per_window $SECONDS_PER_WINDOW --group $GROUP --mode $WANDB_MODE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file $CONFIG_FILE --env $ENV $MODEL_ARGS --use_modalities $AV $LM $EYES
