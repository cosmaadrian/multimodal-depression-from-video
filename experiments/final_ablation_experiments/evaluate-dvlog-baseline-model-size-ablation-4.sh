#!/bin/bash

set -e
cd ../


N_TEMPORAL_WINDOWS=1
PRESENCE_THRESHOLD=0.25
SECONDS_PER_WINDOW=5

ENV="banamar-upv16"
GROUP=dvlog-baseline-model-size-ablation

DEFAULT_ARGS="--group $GROUP --env $ENV --n_temporal_windows $N_TEMPORAL_WINDOWS --seconds_per_window $SECONDS_PER_WINDOW"

#eval on val split
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-1" $DEFAULT_ARGS
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-2" $DEFAULT_ARGS
python evaluate.py --eval_config  configs/eval_configs/eval_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-3" $DEFAULT_ARGS

#eval on test split
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-1" $DEFAULT_ARGS
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-2" $DEFAULT_ARGS
python evaluate.py --eval_config  configs/eval_configs/eval_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name "pt-$PRESENCE_THRESHOLD-spw-$SECONDS_PER_WINDOW-nl-8-nh8-hd32-run-3" $DEFAULT_ARGS
