#!/bin/bash

set -e
cd ../

EPOCHS=250
CONFIG_FILE="configs/baseline_config.yaml"
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
AV_MODALITIES="audio_embeddings face_embeddings"
LANDMARK_MODALITIES="face_landmarks body_landmarks hand_landmarks"
EYE_MODALITIES="gaze_features blinking_features"
ALL_MODALITIES="audio_embeddings face_embeddings face_landmarks body_landmarks hand_landmarks gaze_features blinking_features"

WANDB_MODE=run
ENV="banamar-upv16"
GROUP=dvlog-baseline-model-size-ablation
MODEL_ARGS=""
TRAINING_ARGS="--group $GROUP --mode $WANDB_MODE --env $ENV --config_file $CONFIG_FILE --save_model 1 --epochs $EPOCHS --batch_size $BATCH_SIZE --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS"


for PRESENCE_THRESHOLD in 0.25
do
  for SECONDS_PER_WINDOW in 9
  do
    for RUN_ID in 1 2 3
    do
    DEFAULT_ARGS="--n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW --presence_threshold $PRESENCE_THRESHOLD --use_modalities $AV_MODALITIES"

    WANDB_NAME="pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-$RUN_ID"
    python main.py --run_id $RUN_ID --name $WANDB_NAME $TRAINING_ARGS $DEFAULT_ARGS --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32
    done
  done
done
