#!/bin/bash

# Please set these variables
MPIIGAZE_DIR=
ETH_XGAZE_CONFIG=

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# GAZE TRACKING
python3 ./scripts/feature_extraction/dvlog/extract_gaze_tracking.py --mpiigaze-dir $MPIIGAZE_DIR --eth-xgaze-config-path $ETH_XGAZE_CONFIG --video-dir $VIDEO_DIR --gaze-output-dir $NO_CHUNKED_DIR/gaze_features/ --no-gaze-output-dir $NO_CHUNKED_DIR/no_gaze_idxs/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id gaze_features --no-idxs-id no_gaze_idxs --frame-rate 25
