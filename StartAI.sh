#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AI directory, for example:
# sudo ln -s /home/wally /home/ai
cd /home/ai/AI2

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown, something changed with recent update on 20.04 and 22.04 that makes these not work
usr/bin/pkill -2 -f "AI2.py" > /dev/null 2>&1
#usr/bin/pkill -2 -f "AI.py" > /dev/null 2>&1
sleep 1

# but, make sure it goes away before retrying, these still seem to work but are harsh
/usr/bin/pkill -9 -f "AI2.py" > /dev/null 2>&1
sleep 1

export PYTHONUNBUFFERED=1

# VENV python virtual environment in directory named y8ovv created with:
# python3 -m venv $HOME/y8ovv
# if you used a different VENV name make sym like to avoid needing to edit this
# ln -s yourYoloVenv $HOME/y8ovv


#
# make symlink for camera files in AI2 directory to avoid having to edit this script!
##  ln -s 8UHD.rtsp cameraURL.rtsp
##  ln -s 6onvif.txt cameraURL.txt
#


## CPU initial AI,this will display the live camera windows on the computer screen, can be useful with a laptop
## source ../y8ovv/bin/activate
#python3 AI2.py -nsz -d -y8ovv  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

## this will not display anything, minor speedup, best if AI host is "headless"
#python3 AI2.py -nsz -y8ovv  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

## TPU initial AI, openvion yolo8 detection, use -y8v for CUDA yolo8, -nsz doesn't save zoomed detections
#python3 AI2.py -tpu -nsz -y8ovv  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

## example to tile 4 cameras on a 1920x1080 screen
## algorithym I use:
# define grid, say 2 rows by 3 columns
# fudgeFactor = 8 
# -iw = int(-dw / columns) - fudge factor
# -ih = int(-iw * 9 / 16)
#python3 AI2.py -nsz -d -y8ovv  -dw 1920 -dh 1080 -iw 944 -ih 531 2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

# CPU initial AI detection with yolo8 Cuda verification
##source ../y8cuda/bin/activate
##python3 AI2.py -nsz -d -y8v 2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

# TPU initial AI and yolo8 TPU verification
source ../y8tpu/bin/activate
python3 AI2.py -nsz -d -tpu -y8tpu 2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &
