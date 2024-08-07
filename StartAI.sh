#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AI directory, for example:
# sudo ln -s /home/wally /home/ai
cd /home/ai/AI2

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown
usr/bin/pkill -2 -f "AI2.py" > /dev/null 2>&1
usr/bin/pkill -2 -f "AI.py" > /dev/null 2>&1
sleep 5

# but, make sure it goes away before retrying
/usr/bin/pkill -9 -f "AI2.py" > /dev/null 2>&1
/usr/bin/pkill -9 -f "AI.py" > /dev/null 2>&1
sleep 1

export PYTHONUNBUFFERED=1

cd /home/ai/AI2

### This block is for using CUDA -y8v
# VENV python virtual environment in directory named yolo8 created with:
# python3 -m venv $HOME/yolo8
# needed for CUDA -y8v, not for openvino iGPU yolo8
# for cuda-11.4
###export PATH=/usr/local/cuda-11.4/bin:$PATH
###export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64
###source ../yolo8/bin/activate

### This block is for using -y8ovv yolo8 on openvino 2023.2 iGPU
# VENV python virtual environment in directory named y8ovv created with:
# python3 -m venv $HOME/y8ovv
# needed for -y8ovv, not for CUDA -y8v
source ../y8ovv/bin/activate


# make symlink for camera files in AI2 directory to avoid having to edit this script!
##  ln -s 8UHD.rtsp cameraURL.rtsp
##  ln -s 6onvif.txt cameraURL.txt
#
# TPU initial AI, openvino GPU yolo8 verification, will display the live camera windows on the computer screen, can be useful with a laptop
python3 AI2.py -nsz -tpu -d -y8ovv  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &
#
# CPU initila AI with CUDA yolo8 verification, will not display anything, minor speedup, best if AI host is "headless"
###python3 AI2.py -nsz -y8v  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

