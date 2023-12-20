#!/bin/bash

# Please set these variables
PASE_CONFIG=
PASE_CHCK_PATH=
USE_AUTH_TOKEN=

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# VOICE EMBEDDINGS
python3 ./scripts/feature_extraction/dvlog/extract_wavs.py --csv-path ./data/D-vlog/video_ids.csv --column-video-id video_id --video-dir $VIDEO_DIR --dest-dir $WAV_DIR
python3 ./scripts/feature_extraction/dvlog/extract_audio_pase_embeddings.py --config-file $PASE_CONFIG --checkpoint $PASE_CHCK_PATH --audio-dir $WAV_DIR --audio-embeddings-output-dir $NO_CHUNKED_DIR/audio_pase_embeddings/
python3 ./scripts/feature_extraction/dvlog/extract_voice_activity.py --pretrained-model pyannote/voice-activity-detection --use-auth-token $USE_AUTH_TOKEN --audio-dir ./data/D-vlog/wavs/ --dest-dir $NO_CHUNKED_DIR/audio_activity/
python3 ./scripts/feature_extraction/dvlog/process_voice_activity.py --splits-dir ./data/D-vlog/splits/ --data-root-dir $NO_CHUNKED_DIR --audio-embeddings-dir audio_pase_embeddings --audio-activity-dir audio_activity --dest-dir no_voice_idxs
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --dest-dir $DATA_DIR --modality-id audio_pase_embeddings --no-idxs-id no_voice_idxs --frame-rate 100
