#!/bin/bash

set -e

## DAIC-WOZ ##
bash ./daicwoz/train-baseline-daicwoz-modality-ablation.sh
bash ./daicwoz/evaluate-baseline-daicwoz-modality-ablation.sh

bash ./daicwoz/train-perceiver-daicwoz-modality-ablation.sh
bash ./daicwoz/evaluate-perceiver-daicwoz-modality-ablation.sh

## D-VLOG ##
bash ./dvlog/train-baseline-dvlog-modality-ablation.sh
bash ./dvlog/evaluate-baseline-dvlog-modality-ablation.sh

bash ./dvlog/train-perceiver-dvlog-modality-ablation.sh
bash ./dvlog/evaluate-perceiver-dvlog-modality-ablation.sh

## E-DAIC ##
bash ./edaic/train-baseline-edaic-modality-ablation.sh
bash ./edaic/evaluate-baseline-edaic-modality-ablation.sh

bash ./edaic/train-perceiver-edaic-modality-ablation.sh
bash ./edaic/evaluate-perceiver-edaic-modality-ablation.sh

## ORIGINAL D-VLOG + NEW SPLIT ##
bash ./original-dvlog-new-split/train-baseline-original-dvlog-new-split-modality-ablation.sh
bash ./original-dvlog-new-split/evaluate-baseline-original-dvlog-new-split-modality-ablation.sh

bash ./original-dvlog-new-split/train-perceiver-original-dvlog-new-split-modality-ablation.sh
bash ./original-dvlog-new-split/evaluate-perceiver-original-dvlog-new-split-modality-ablation.sh
