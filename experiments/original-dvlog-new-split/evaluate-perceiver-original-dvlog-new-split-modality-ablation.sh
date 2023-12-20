#!/bin/bash

set -e
cd ..

BATCH_SIZE=8
SECONDS_PER_WINDOW=9

GROUP=perceiver-original-dvlog-new-split-modality-ablation

EVAL_TEST_CONFIG="configs/eval_configs/eval_original_dvlog_new_split_test_config.yaml"
EVAL_VAL_CONFIG="configs/eval_configs/eval_original_dvlog_new_split_val_config.yaml"
ENV="reading-between-the-frames"

# EVALUATING ON TEST SET

# AV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# EVALUATING ON VALIDATION SET

# AV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name av-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
