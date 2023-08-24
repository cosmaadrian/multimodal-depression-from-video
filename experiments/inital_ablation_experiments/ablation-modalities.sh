#!/bin/bash

set -e
cd ..

WANDB_MODE=dryrun
BATCH_SIZE=8
EPOCHS=500
GROUP=baseline-modalities

LANDMARKS='face_landmarks body_landmarks hand_landmarks'
EYES='gaze_features blinking_features'
AUDIO_VISUAL='audio_embeddings face_embeddings'
ALL_MODALITIES='audio_embeddings face_embeddings face_landmarks body_landmarks hand_landmarks gaze_features blinking_features'

# Audio-Visual [x]

python main.py --run_id 1 --presence_threshold 0.5 --name av-lm-eyes-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
exit -1

# Audio-visual + landmarks + eyes
python main.py --run_id 1 --presence_threshold 0.5 --name av-lm-eyes-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 2 --presence_threshold 0.5 --name av-lm-eyes-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES
python main.py --run_id 3 --presence_threshold 0.5 --name av-lm-eyes-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $ALL_MODALITIES

# Audio-visual + landmarks
python main.py --run_id 1 --presence_threshold 0.5 --name av-lm-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
python main.py --run_id 2 --presence_threshold 0.5 --name av-lm-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS
python main.py --run_id 3 --presence_threshold 0.5 --name av-lm-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $LANDMARKS

# Audio-visual + eyes
python main.py --run_id 1 --presence_threshold 0.5 --name av-eyes-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 2 --presence_threshold 0.5 --name av-eyes-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES
python main.py --run_id 3 --presence_threshold 0.5 --name av-eyes-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $AUDIO_VISUAL $EYES

# Landmarks + eyes
python main.py --run_id 1 --presence_threshold 0.5 --name lm-eyes-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS $EYES
python main.py --run_id 2 --presence_threshold 0.5 --name lm-eyes-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS $EYES
python main.py --run_id 3 --presence_threshold 0.5 --name lm-eyes-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS $EYES

# Landmarks (face, hands, body)
python main.py --run_id 1 --presence_threshold 0.5 --name lm-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS
python main.py --run_id 2 --presence_threshold 0.5 --name lm-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS
python main.py --run_id 3 --presence_threshold 0.5 --name lm-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $LANDMARKS

# Eyes (blinking + gaze)
python main.py --run_id 1 --presence_threshold 0.5 --name eyes-run-1 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $EYES
python main.py --run_id 2 --presence_threshold 0.5 --name eyes-run-2 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $EYES
python main.py --run_id 3 --presence_threshold 0.5 --name eyes-run-3 --seconds_per_window 5 --group $GROUP --mode $WANDB_MODE --epochs $EPOCHS --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities $EYES
