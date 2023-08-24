#!/bin/bash

set -e
cd ..


# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.0001-run-1 --group perceiver-cosinelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.0001-run-2 --group perceiver-cosinelr --env banamar-upv16
# was not run python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.0001-run-3 --group perceiver-cosinelr --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.00001-run-1 --group perceiver-cosinelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.00001-run-2 --group perceiver-cosinelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cosinelr --checkpoint_kind best --name cosinelr-0.00001-run-3 --group perceiver-cosinelr --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.00001-run-1 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.00001-run-2 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.00001-run-3 --group perceiver-cyclelr --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.0001-run-1 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.0001-run-2 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.0001-run-3 --group perceiver-cyclelr --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.001-run-1 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.001-run-2 --group perceiver-cyclelr --env banamar-upv16
# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir perceiver-cyclelr --checkpoint_kind best --name cyclelr-lr-0.001-run-3 --group perceiver-cyclelr --env banamar-upv16

# python evaluate.py --eval_config  configs/eval_config.yaml --output_dir balanced-perceiver-cosinelr-n-windows --checkpoint_kind best --name cosinelr-0.00001-n-windows-1-run-1 --group balanced-perceiver-cosinelr-n-windows --env banamar-upv16
python evaluate.py --eval_config  configs/eval_config.yaml --output_dir balanced-perceiver-cosinelr-n-windows --checkpoint_kind best --name cosinelr-0.00001-n-windows-2-run-1 --n_temporal_windows 1 --group balanced-perceiver-cosinelr-n-windows --env banamar-upv16
python evaluate.py --eval_config  configs/eval_config.yaml --output_dir balanced-perceiver-cosinelr-n-windows --checkpoint_kind best --name cosinelr-0.00001-n-windows-4-run-1 --n_temporal_windows 1 --group balanced-perceiver-cosinelr-n-windows --env banamar-upv16
python evaluate.py --eval_config  configs/eval_config.yaml --output_dir balanced-perceiver-cosinelr-n-windows --checkpoint_kind best --name cosinelr-0.00001-n-windows-6-run-1 --n_temporal_windows 1 --group balanced-perceiver-cosinelr-n-windows --env banamar-upv16