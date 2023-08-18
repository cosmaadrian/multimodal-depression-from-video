#!/bin/bash

set -e
cd ..

WANDB_MODE=dryrun
GROUP=baseline-big-model-window-size-ablation-2


python evaluate.py --eval_config  configs/eval_config.yaml --output_dir $GROUP --checkpoint_kind best --name window-size-10-run-1 --group $GROUP --mode $WANDB_MODE --env banamar-upv16