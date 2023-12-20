#!/bin/bash

# Please set these variables
INSTBLINK_CONFIG=
INSTBLINK_CHCK_PATH=

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# BLINKING PATTERNS
python3 ./scripts/feature_extraction/dvlog/extract_blinking_patterns.py --config $INSTBLINK_CONFIG --checkpoint $INSTBLINK_CHCK_PATH --video-root $VIDEO_DIR --dest-root $NO_CHUNKED_DIR/blinking_patterns/ --dest-no-idxs-root $NO_CHUNKED_DIR/no_blink_idxs/ --tmp-root $NO_CHUNKED_DIR/video_frames/
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id blinking_patterns --no-idxs-id no_blink_idxs --frame-rate 25
