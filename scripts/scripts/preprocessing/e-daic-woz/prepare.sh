#!/bin/bash

set -e

# python ./scripts/prepare_bow_egemaps.py
# python ./scripts/prepare_bow_mfcc.py
# python ./scripts/prepare_bow_pose_gaze_aus.py
# python ./scripts/prepare_cnn_resnet.py
python ./scripts/prepare_cnn_vgg.py
python ./scripts/prepare_densenet.py
python ./scripts/prepare_egemaps.py
python ./scripts/prepare_mfcc.py
python ./scripts/prepare_pose_gaze_aus.py
python ./scripts/prepare_vgg16.py
