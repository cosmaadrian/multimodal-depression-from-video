#!/bin/bash

# -- installing packages
pip install  https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
pip install pandas
pip install tqdm
pip install joblib
conda install -c conda-forge phantomjs

# -- git lfs installation
conda install -c conda-forge git-lfs
git lfs install

# -- the following toolkits include the \
# installation of other useful packages, such as PyTorch
cd ./tools/

# -- uncomment to install the face detection toolkit
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

# -- uncomment to install the face alignment toolkit
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..

# -- uncomment to install the body and hand pose landmark estimators
pip install -q mediapipe==0.10.0
mkdir ./landmarkers/
wget -O ./landmarkers/body_pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -O ./landmarkers/hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cd ../
