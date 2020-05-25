#!/usr/bin/env bash

export CONDA_ENV_NAME=pix2surf
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.6.9

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

conda install pytorch==1.0.1 torchvision==0.2.1 -c pytorch
conda install pillow==6.2.1
conda install numpy==1.14.2
pip install opencv-python
pip install tensorboardX==1.6
pip install gdown