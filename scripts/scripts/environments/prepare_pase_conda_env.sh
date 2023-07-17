#!/bin/bash

conda create -y -n pase+ python=3.7
conda activate pase+

conda install -c anaconda cudatoolkit=10.1
cd ./tools/
git clone https://github.com/santi-pdp/pase.git
cd ./pase/
pip install -r requirements.txt
pip install torchvision==0.5.0

python setup.py install

# Download model checkpoint!
# https://drive.google.com/file/d/1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW/view

cd ../../
