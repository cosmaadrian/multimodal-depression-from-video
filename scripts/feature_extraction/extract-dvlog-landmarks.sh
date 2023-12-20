#!/bin/bash

# Please set these variables
BODY_LANDMARKER_PATH=
HAND_LANDMARKER_PATH

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# FACE LANDMARKS
python3 ./scripts/feature_extraction/dvlog/extract_faces_and_landmarks.py --video-dir $VIDEO_DIR --landmarks-output-dir $NO_CHUNKED_DIR/face_landmarks/ --no-faces-output-dir $NO_CHUNKED_DIR/no_face_idxs/ --faces-output-dir $NO_CHUNKED_DIR/faces/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id face_landmarks --no-idxs-id no_face_idxs --frame-rate 25

# BODY LANDMARKS
python3 ./scripts/feature_extraction/dvlog/extract_body_pose_landmarks.py --pose-landmarker-path $BODY_LANDMARKER_PATH --video-dir $VIDEO_DIR --landmarks-output-dir $NO_CHUNKED_DIR/body_landmarks/ --no-landmarks-output-dir $NO_CHUNKED_DIR/no_body_idxs/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id body_landmarks --no-idxs-id no_body_idxs --frame-rate 25

# HAND LANDMARKS
python3 ./scripts/feature_extraction/dvlog/extract_hand_landmarks.py --hand-landmarker-path $HAND_LANDMARKER_PATH --video-dir $VIDEO_DIR --landmarks-output-dir $NO_CHUNKED_DIR/hand_landmarks/ --no-landmarks-output-dir $NO_CHUNKED_DIR/no_hand_idxs/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id hand_landmarks --no-idxs-id no_hand_idxs --frame-rate 25
