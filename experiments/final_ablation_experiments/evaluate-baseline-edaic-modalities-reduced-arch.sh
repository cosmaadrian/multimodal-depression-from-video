#!/bin/bash

set -e
cd ..

WANDB_MODE=run

BATCH_SIZE=8
SECONDS_PER_WINDOW=6

GROUP=hope-baseline-edaic-modalities-reduced-arch
A_MODALITIES="edaic_audio_mfcc edaic_audio_egemaps"
V_MODALITIES="edaic_video_cnn_resnet edaic_video_pose_gaze_aus"
AV_MODALITIES="edaic_audio_mfcc edaic_audio_egemaps edaic_video_cnn_resnet edaic_video_pose_gaze_aus"

# EVALUATING ON TEST SET
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16


# EVALUATING ON VALIDATION SET
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_edaic_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
