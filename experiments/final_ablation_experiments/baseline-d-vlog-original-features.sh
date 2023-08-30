#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=64
EPOCHS=250
GROUP=baseline-d-vlog-original

# python main.py --run_id 1 --name baseline-run-1 --save_model 1 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_config.yaml --dataset original-d-vlog --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 2 --name baseline-run-2 --save_model 1 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_config.yaml --dataset original-d-vlog --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 3 --name baseline-run-3 --save_model 1 --n_temporal_windows 1 --seconds_per_window 10 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_config.yaml --dataset original-d-vlog --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors

python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --name baseline-run-1 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --name baseline-run-2 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_val_original_dvlog_config.yaml --name baseline-run-3 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16

python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --name baseline-run-1 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --name baseline-run-2 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config  configs/eval_configs/eval_test_original_dvlog_config.yaml --name baseline-run-3 --n_temporal_windows 1 --seconds_per_window 10 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
