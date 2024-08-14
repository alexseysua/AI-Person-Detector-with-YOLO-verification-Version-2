# AI-Person-Detector-with-YOLO-verification-Version-2
This is "Version 2" of the repo: https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification

These are my notes for a virgin installation of Ubuntu 22.04 and installation of OpenVINO 2024 release 
for using Intel integrated GPU for yolo verification.  Motivated by 
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
Instructions for using CUDA will be given below, but you will still need OpenVINO for the MobilenetSSD_v2 initial AI if not using the Coral TPU, whos instructions are also given below.

### 3) Clone this repo
Rename the directory to AI2, otherwise you'll need to edit the node-red scripts to account for the different names, then activate the virtual environment:
### source y8ovv/bin/activate
### cd AI2
Next we need to tell the python code how to talk to the cameras. Two types of cameras are supported, Onvif and RTSP.  Onvif is an overly complited "standard" that is rarely implimented fully or correctly but if you can retrieve an image with an HTTP request it is an Onvif camera for our purposes.  RTSP opens a connection on port 554 and returns a video stream, these are the most common type of cameras.  Be aware that RING, SimplySafe, Arlo, Blink etc. don't generally allow direct access to the video streams or still images.  Also the low end securtiy DVRs like Swann, NightOwl also usually lack support for RTSP streams.  Before you buy, make sure the camera or DVRs you are condidering support RTSP streams and or "Onvif snapshots".

To tell the python code how to use your cameras you need to create a cameraURL.txt file for Onvif cameras or a cameraURL.rtsp file for RTSP cameras.  A cameraURL.txt file should one line per camera containing the HTTP URL to retrieve and image and after a space contain the optional name for what the camera is viewing.
A cameraURL.txt file for two cameras would look like this:
```
http://192.168.2.144:85/images/snapshot.jpg Driveway
http://admin:password@192.168.2.197:80/tmpfs/auto.jpg  Garage
```
Note than some cameras don't require any authorization to return the image, others do and the URLs can be very different format.  It is easy to test your URL simpley by entering into a web browser and you should see and image after connecting and a fresh imiage after refreshing the page.
A cameraURL.rtsp file should look like this:
```
rtsp://admin:passwd@192.168.2.28:554/cam/realmonitor?channel=1&subtype=0  MailBox
rtsp://admin:passwd%@192.168.2.28:554/cam/realmonitor?channel=7&subtype=0  DriveWay
rtsp://admin:passwd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Garage
rtsp://admin:PassWd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Patio
rtsp://admin:PaSwrd@192.168.2.196:554/h264Preview_01_main FronYard
rtsp://192.168.2.77:554/stream0?username=admin&password=CE04588A3231F95BEE71848F5958CF4E BackYard
```
The The IP addresses will be what your router assigns, my example shows two security DVRs at xxx.xxx.xxx.28 and xxx.xxx.xxx.49 and two IP or "netcams" the last netcam generates the password as part of the Onvif setup and can be "tricky" to figure out, but they are not generally unique.  Best way I know to test your RTSP URLs is with VLC and "Open Network" and enter the RTSP URL string.  Main negative of RTSP streams is the latency is typically 2-4 seconds behind "real time".

Once the camera URLs are specified (both types of camera files are allowed) we can run a quick test, make sure the virtual environment is active and open up a command window (terminal):
```
python AI2.py -d
```
This will start a thread for each video camaera and an OpenVINO CPU AI thread and display live results on the screen.  Here is an image of it running on a Lenovo IdeaPad i3 laptop doing six Onvif cameras:
![IdeaPad](https://github.com/user-attachments/assets/647f3c54-a265-406a-833d-0ef1a2883eec)  Notifications and the housekeeping functions are done with node-red which we will install next.



