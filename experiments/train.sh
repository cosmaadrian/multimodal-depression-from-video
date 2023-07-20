#!/bin/bash

set -e
cd ..

# python main.py --config_file configs/perceiver_config.yaml --debug 1 --env banamar-upv16
python main.py --config_file configs/baseline_config.yaml --debug 1 --env banamar-upv16
