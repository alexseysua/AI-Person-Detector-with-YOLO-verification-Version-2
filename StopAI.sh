#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AIdev directory, for example:
# sudo ln -s /home/wally /home/AI
cd /home/ai/AI2

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown but recent ~6AUG2024 Ubuntu update seems to have broken signalling with pkill
usr/bin/pkill -15 -f "AI2.py" > /dev/null 2>&1
usr/bin/pkill -2 -f "AI2.py" > /dev/null 2>&1
usr/bin/pkill -2 -f "AI.py" > /dev/null 2>&1
sleep 6

# but, make sure it goes away before retrying
#/usr/bin/pkill -9 -f "AI2.py" > /dev/null 2>&1
/usr/bin/pkill -9 -f "AI.py" > /dev/null 2>&1
sleep 1

