#!/bin/bash

set -e
cd ..

# python main.py --config_file configs/perceiver_config.yaml --env banamar-upv16
# --mode run
# --mode dryrun

# python main.py --name baseline-model --group test --mode dryrun --config_file configs/baseline_config.yaml --env banamar-upv16
python main.py --batch_size 32 --debug 1 --name perceiver-model --group test --mode dryrun --config_file configs/perceiver_config.yaml --env banamar-upv16
