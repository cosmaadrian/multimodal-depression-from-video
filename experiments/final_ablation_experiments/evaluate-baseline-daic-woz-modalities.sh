#!/bin/bash

set -e
cd ..

WANDB_MODE=run

SECONDS_PER_WINDOW=7

GROUP=baseline-daic-woz-modalities+class-weights
A_MODALITIES="daic_audio_covarep daic_audio_formant"
V_MODALITIES="daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"
AV_MODALITIES="daic_audio_covarep daic_audio_formant daic_facial_3d_landmarks daic_facial_hog daic_facial_aus daic_gaze daic_head_pose"

# EVALUATING ON TEST SET
BATCH_SIZE=8
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

BATCH_SIZE=4
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_test_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16


# EVALUATING ON VALIDATION SET
BATCH_SIZE=8
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audio-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name video-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16

BATCH_SIZE=4
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-2 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-3 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-4 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_daicwoz_val_config.yaml --output_dir $GROUP --checkpoint_kind best --name audiovisual-5 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env banamar-upv16
