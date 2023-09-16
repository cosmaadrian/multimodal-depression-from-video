#!/bin/bash

set -e
cd ..

WANDB_MODE=run

BATCH_SIZE=8
SECONDS_PER_WINDOW=9
PRESENCE_THRESHOLD=0.5

GROUP=perceiver-original-dvlog
ALL_MODALITIES='orig_audio_descriptors orig_face_landmarks'

# EVALUATING ON TEST SET
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# EVALUATING ON VALIDATION SET
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16