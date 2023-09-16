#!/bin/bash

set -e
cd ..

WANDB_MODE=run

BATCH_SIZE=8
SECONDS_PER_WINDOW=9
PRESENCE_THRESHOLD=0.5

GROUP=perceiver-ablation-modalities
LANDMARKS='face_landmarks body_landmarks hand_landmarks'
EYES='gaze_features blinking_features'
AUDIO_VISUAL='audio_embeddings face_embeddings'
ALL_MODALITIES='audio_embeddings face_embeddings face_landmarks body_landmarks hand_landmarks gaze_features blinking_features'

# EVALUATING ON TEST SET
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-6 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-7 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-8 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-9 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-10 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16


# EVALUATING ON VALIDATION SET
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm+eyes-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+lm-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-6 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-7 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-8 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-9 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av+eyes-run-10 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
