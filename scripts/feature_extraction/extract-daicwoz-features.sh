#!/bin/bash

set -e

NO_CHUNKED_DIR=./data/DAIC-WOZ/no-chunked/

python ./scripts/feature_extraction/daicwoz/untar_data.py --root-dir ./data/DAIC-WOZ/backup/ --dest-dir ./data/DAIC-WOZ/original_data/
python ./scripts/feature_extraction/daicwoz/get_no_idxs.py --data-dir ./data/DAIC-WOZ/original_data/ --dest-dir ./data/DAIC-WOZ/no-chunked/

# COVAREP
python ./scripts/feature_extraction/daicwoz/prepare_covarep.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id audio_covarep --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_covarep --no-idxs-id no_voice_idxs --dest-dir ./data/DAIC-WOZ/data/

# FORMANTS
python ./scripts/feature_extraction/daicwoz/prepare_formant.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id audio_formant --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_formant --no-idxs-id no_voice_idxs --dest-dir ./data/DAIC-WOZ/data/

# 68 3D FACIAL LANDMARKS
python ./scripts/feature_extraction/daicwoz/prepare_clnf_features3D.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id facial_3d_landmarks --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id facial_3d_landmarks --no-idxs-id no_face_idxs --dest-dir ./data/DAIC-WOZ/data/

# FACIAL ACTION UNITS
python ./scripts/feature_extraction/daicwoz/prepare_clnf_aus.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id facial_aus --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id facial_aus --no-idxs-id no_face_idxs --dest-dir ./data/DAIC-WOZ/data/

# GAZE
python ./scripts/feature_extraction/daicwoz/prepare_clnf_gaze.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id gaze --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id gaze --no-idxs-id no_face_idxs --dest-dir ./data/DAIC-WOZ/data/

# HEAD POSE
python ./scripts/feature_extraction/daicwoz/prepare_clnf_pose.py --src-root ./data/DAIC-WOZ/original_data/ --modality-id head_pose --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/daicwoz/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id head_pose --no-idxs-id no_face_idxs --dest-dir ./data/DAIC-WOZ/data/

