#!/bin/bash

# Please set these variables
EMONET_CHCK_PATH=

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# FACE EMBEDDINGS
python3 ./scripts/feature_extraction/dvlog/extract_face_emonet_features.py --checkpoint $EMONET_CHCK_PATH --faces-dir $NO_CHUNKED_DIR/faces/ --face-embeddings-output-dir $NO_CHUNKED_DIR/face_emonet_embeddings/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id face_emonet_embeddings --no-idxs-id no_face_idxs --frame-rate 25
