#!/bin/bash

set -e

NO_CHUNKED_DIR=./data/E-DAIC/no-chunked/

python ./scripts/feature_extraction/edaic/untar_data.py --root-dir ./data/E-DAIC/backup/ --dest-dir ./data/E-DAIC/original_data/
python ./scripts/feature_extraction/edaic/get_no_idxs.py --data-dir ./data/E-DAIC/original_data/ --dest-dir ./data/E-DAIC/no-chunked/

# MFCC
python ./scripts/feature_extraction/edaic/prepare_mfcc.py --src-root ./data/E-DAIC/original_data/ --modality-id audio_mfcc --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_mfcc --no-idxs-id no_voice_idxs --dest-dir ./data/E-DAIC/data/

# eGeMaps
python ./scripts/feature_extraction/edaic/prepare_egemaps.py --src-root ./data/E-DAIC/original_data/ --modality-id audio_egemaps --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_egemaps --no-idxs-id no_voice_idxs --dest-dir ./data/E-DAIC/data/

# cnn face features
python ./scripts/feature_extraction/edaic/prepare_cnn_resnet.py --src-root ./data/E-DAIC/original_data/ --modality-id video_cnn_resnet --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id video_cnn_resnet --no-idxs-id no_face_idxs --dest-dir ./data/E-DAIC/data/

# gaze pose aus
python ./scripts/feature_extraction/edaic/prepare_pose_gaze_aus.py --src-root ./data/E-DAIC/original_data/ --modality-id video_pose_gaze_aus --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id video_cnn_resnet --no-idxs-id no_face_idxs --dest-dir ./data/E-DAIC/data/

