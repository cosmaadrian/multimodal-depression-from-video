#!/bin/bash

set -e
cd ..

WANDB_MODE=run

BATCH_SIZE=8
SECONDS_PER_WINDOW=6

GROUP=perceiver-daicwoz-ablation-modalities
A_MODALITIES="daic_audio_covarep daic_audio_formant"
V_MODALITIES="daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"
AV_MODALITIES="daic_audio_covarep daic_audio_formant daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"

# EVALUATING ON TEST SET
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16


# EVALUATING ON VALIDATION SET
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
