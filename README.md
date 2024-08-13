# AI-Person-Detector-with-YOLO-verification-Version-2
This is "Version 2" of the repo: https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification

These are my notes for a virgin installation of Ubuntu 22.04 and installation of OpenVINO 2024 release 
for using Intel integraded GPU for yolo verification.  Motivated by 
https://igor.technology/installing-coral-usb-accelearator-python-3-10-ubuntu-22/
Which allows using the TPU on system python 3.10 eliminating the need for virtual environments for the basic AI,
although I still strongly recommend using virtual environments since breaking old code has never been much of a consideration for OpenVINO.


## 0) Install Ubuntu 22.04
I used Ubuntu-Mate.  Customize it as you see fit.

If you are new to Linux, first thing after installing Ubuntu-Mate, go through the "Welcome" tutorial to learn the basics and then go to the "Control Center" and open "MATE Tweak" and from the sidebar select "Panel".  If you choose "Redmon" from the dropdown you'll get a destop that resembles Windows, if you choose "Cupetino" you'll get a Mac-like desktop.  If you can tolerate the Ubuntu "Unity" Desktop, fine, but I can't, and if coming from Windows or Mac be prepared for "eveything you know is wrong" when it comes to using the desktop UI.

All my Python code, OpenVINO, node-red, Coral TPU drivers (if you need/want them) and CUDA (if you are using nVidia GPU instead of Intel) should be available on Windows, but I've not tested it.  But you'll lose the "housekeeping" functionality in some shell scripts that are called via node-red exec nodes in my minimal webbased "controller".  Minimal is a design feature, I wanted a "set and forget" appliance that runs 24/7/365 and is controlled by your "home automation" or a web browser to set one of three modes of operation.  "Notify" mode send Email alerts,  "Audio" uses "Espeak" speech synthsizer to announce what camera a person has been detected on, and Idle just saves detection without any nodifications.  If you manage to run it on Windows please submit your instructions so I can add them here.

Use your username where I have "ai" and your hostname where I have "YouSeeX1".
To avoid having to edit the scripts used by node-red make a sym link in /home:
### sudo ln -s /home/YourUserName /home/ai

## 1) Install needed packages
I like loging in remotely over via ssh, as running "headless" is a design goal, but it can all be done with a termenal window as well. OpenSSH is not installed by default so either install "OpenSSH" using "Control Center Software Botique" or in a terminal window do:
### sudo apt install ssh
Now in a terminl window or your remote shell login, install these extra packages with:
### sudo apt install git curl samba python3-dev python3-pip python3.10-venv espeak mosquitto mosquitto-dev mosquitto-clients cmake

## 2) Create Python Virtual environment and install the needed python modules
Note Conda works well too, and I prefer it if you need a dirrerent python version than the system python3 but Conda  has issues veing launched via a shell script via a node-red exec node.  If you know a solution, please send it to me.  I'm not very good with GitHub so you may need to help me with doing "pull requests" if it is not something you can just Email to me.
Now create a virtual environment to to use with OpenVINO and YOLO8, named y8ovv.
### python3 -m venv y8ovv
Next "activate" the virtual environment:
### source y8ovv/bin/activate
Note that the prompt changes from: ai@YouSeeX1: to: (y8ovv) ai@YouSeeX1:
Use pip to install the needed modules:
### pip install imutils paho-mqtt requests
### pip install "openvino>=2024.2.0" "nncf>=2.9.0"
### pip install "torch>=2.1" "torchvision>=0.16" "ultralytics==8.2.24" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
If not using an Nvidia GPU and Cuda install the Intel GPU driver:
### sudo apt-get install intel-opencl-icd
Add the ai user (your login if you didn't use ai as your username at installation) to the render group:
Make sure the following command is being done as user, not as sudo -i
### sudo adduser $USER render
Now log out and login or reboot the system. GPU doesn't work if you don't do this!
Instruction for using CUDA will be given below, but you will still need OpenVINO for the MobilenetSSD_v2 initial AI if not using the Coral TPU, whos instructions are also given below.

