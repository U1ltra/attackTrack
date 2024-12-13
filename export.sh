#!/bin/sh
cd ~/Documents/github/pysot
export PYTHONPATH=$PWD:$PYTHONPATH

cd ~/Documents/github/SiamMask
export SiamMask=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH

cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH

cd ~/Documents/github/attackTrack
export DISPLAY=attackSer1:10.0
# export CUDA_VISIBLE_DEVICES=1

# ls -l /tmp/.X11-unix/
export DISPLAY=:1

# conda install numpy
# conda install pytorch=1.7.1 torchvision=0.8.2 -c pytorch
# pip install kornia==0.5.0