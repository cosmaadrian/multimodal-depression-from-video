#!/bin/bash

conda install -c anaconda cudatoolkit=10.1

mkdir ./feature_extractors/
cd ./feature_extractors/

git clone https://github.com/santi-pdp/pase.git
cd ./pase/
pip install -r requirements.txt
pip install torchvision==0.5.0

python setup.py install

pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW/view

cd ../../
