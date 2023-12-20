#!/bin/bash

pip install pandas
pip install tqdm
pip install joblib
conda install -c conda-forge phantomjs

conda install -c conda-forge git-lfs
git lfs install

mkdir ./feature_extractors/
cd ./feature_extractors/

git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..

pip install -q mediapipe==0.10.0
wget -O ./body_pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -O ./hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cd ..
