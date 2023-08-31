#!/bin/bash

set -e
cd ..

EPOCHS=250
TRAINER="classification"
SCHEDULER="cosine"
MAX_LR=0.001
BATCH_SIZE=8
N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.5
USE_MODALITIES="audio_embeddings face_embeddings"

WANDB_MODE=run
ENV="banamar-upv16"
GROUP=baseline-ws-ablation-final
DEFAULT_ARGS="--group $GROUP --mode $WANDB_MODE --env $ENV --config_file configs/baseline_config.yaml --save_model 1 --trainer $TRAINER --scheduler $SCHEDULER --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --n_temporal_windows $N_TEMPORAL_WINDOWS --presence_threshold $PRESENCE_THRESHOLD --epochs $EPOCHS --batch_size $BATCH_SIZE --use_modalities $USE_MODALITIES"


##################################
# Evaluating on the validation set
##################################

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-1 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-2 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-1-run-3 --n_temporal_windows 1 --seconds_per_window 1 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-1 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-2 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-2-run-3 --n_temporal_windows 1 --seconds_per_window 2 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-1 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-2 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-3 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-1 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-2 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-3 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-1 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-2 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-3 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-1 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-2 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-3 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-1 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-2 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-3 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV

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

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-1 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-2 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-3-run-3 --n_temporal_windows 1 --seconds_per_window 3 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-1 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-2 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-4-run-3 --n_temporal_windows 1 --seconds_per_window 4 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-1 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-2 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-5-run-3 --n_temporal_windows 1 --seconds_per_window 5 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-1 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-2 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-6-run-3 --n_temporal_windows 1 --seconds_per_window 6 --group $GROUP --env $ENV

# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-1 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-2 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV
# python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-7-run-3 --n_temporal_windows 1 --seconds_per_window 7 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-1 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-2 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-8-run-3 --n_temporal_windows 1 --seconds_per_window 8 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-1 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-2 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-9-run-3 --n_temporal_windows 1 --seconds_per_window 9 --group $GROUP --env $ENV

python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-1 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-2 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-3 --n_temporal_windows 1 --seconds_per_window 10 --group $GROUP --env $ENV