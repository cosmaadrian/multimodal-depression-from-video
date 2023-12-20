#!/bin/bash

set -e
cd ..

BATCH_SIZE=8
SECONDS_PER_WINDOW=6

GROUP=perceiver-edaic-modality-ablation

EVAL_TEST_CONFIG="configs/eval_configs/eval_edaic_test_config.yaml"
EVAL_VAL_CONFIG="configs/eval_configs/eval_edaic_val_config.yaml"
ENV="reading-between-the-frames"

# EVALUATING ON TEST SET

# A
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# V
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# AV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_TEST_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# EVALUATING ON VALIDATION SET

# A
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# V
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

# AV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV
