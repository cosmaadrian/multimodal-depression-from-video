#!/bin/bash

set -e
cd ..

WANDB_MODE=dryrun
BATCH_SIZE=8
GROUP=baseline-big-model-window-size-ablation-2


python evaluate.py --eval_config  configs/eval_config.yaml --output_dir $GROUP --checkpoint_kind best --run_id 1 --name window-size-10-run-1 --seconds_per_window 10 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --config_file configs/baseline_config.yaml --env banamar-upv16 --use_modalities audio_embeddings face_embeddings