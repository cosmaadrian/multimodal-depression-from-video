#!/bin/bash

set -e
cd ..

WANDB_MODE=run
BATCH_SIZE=8
EPOCHS=200
GROUP=baseline-original-dvlog-new-split

# python main.py --run_id 1 --name av-run-1 --save_model 1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --n_temporal_windows 1 --seconds_per_window 9 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 2 --name av-run-2 --save_model 1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --n_temporal_windows 1 --seconds_per_window 9 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 3 --name av-run-3 --save_model 1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --n_temporal_windows 1 --seconds_per_window 9 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 4 --name av-run-4 --save_model 1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --n_temporal_windows 1 --seconds_per_window 9 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors
# python main.py --run_id 5 --name av-run-5 --save_model 1 --model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32 --n_temporal_windows 1 --seconds_per_window 9 --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE  --scheduler_args.max_lr 0.001 --scheduler_args.end_epoch $EPOCHS --config_file configs/original_dvlog_new_split_config.yaml --dataset original-d-vlog-new-split --env banamar-upv16 --use_modalities orig_face_landmarks orig_audio_descriptors

python evaluate.py --eval_config configs/eval_configs/eval_val_original_dvlog_new_split_config.yaml --name av-run-1 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config configs/eval_configs/eval_val_original_dvlog_new_split_config.yaml --name av-run-2 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config configs/eval_configs/eval_val_original_dvlog_new_split_config.yaml --name av-run-3 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
# python evaluate.py --eval_config configs/eval_configs/eval_val_original_dvlog_new_split_config.yaml --name av-run-4 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
# python evaluate.py --eval_config configs/eval_configs/eval_val_original_dvlog_new_split_config.yaml --name av-run-5 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16

python evaluate.py --eval_config configs/eval_configs/eval_test_original_dvlog_new_split_config.yaml --name av-run-1 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config configs/eval_configs/eval_test_original_dvlog_new_split_config.yaml --name av-run-2 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
python evaluate.py --eval_config configs/eval_configs/eval_test_original_dvlog_new_split_config.yaml --name av-run-3 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
# python evaluate.py --eval_config configs/eval_configs/eval_test_original_dvlog_new_split_config.yaml --name av-run-4 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
# python evaluate.py --eval_config configs/eval_configs/eval_test_original_dvlog_new_split_config.yaml --name av-run-5 --n_temporal_windows 1 --seconds_per_window 9 --output_dir $GROUP --group $GROUP --batch_size $BATCH_SIZE --env banamar-upv16
