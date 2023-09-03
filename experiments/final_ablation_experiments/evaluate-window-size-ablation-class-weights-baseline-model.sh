#!/bin/bash

set -e
cd ..

EPOCHS=250
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.25
USE_MODALITIES="audio_embeddings face_embeddings"

ENV="banamar-upv16"
GROUP=baseline-ws-class-weights-ablation-final

##################################
# Evaluating on the validation set
##################################

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-1 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-2 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-3 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-1 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-2 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-3 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-1 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-2 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-3 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-1 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-2 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-3 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-1 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-2 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-3 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-1 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-2 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-3 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-1 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-2 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-3 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-1 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-2 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-3 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-1 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-2 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-3 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-1 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-2 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-3 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV


##################################
# Evaluating on the test set
##################################

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-1 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-2 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-3 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-1 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-2 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-3 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-1 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-2 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-3 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-1 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-2 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-3 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-1 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-2 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-3 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-1 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-2 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-3 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-1 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-2 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-3 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-1 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-2 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-3 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-1 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-2 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-3 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-1 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-2 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-3 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV