#!/bin/bash

set -e
cd ..

python main.py --config_file configs/perceiver-config.yaml --debug 1
