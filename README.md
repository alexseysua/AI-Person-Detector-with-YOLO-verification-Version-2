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
```
sudo ln -s /home/YourUserName /home/ai
```

## 1) Install needed packages
I like loging in remotely over via ssh, as running "headless" is a design goal, but it can all be done with a termenal window as well. OpenSSH is not installed by default so either install "OpenSSH" using "Control Center Software Botique" or in a terminal window do:
```
sudo apt install ssh
```
Now in a terminl window or your remote shell login, install these extra packages with:
```
sudo apt install git curl samba python3-dev python3-pip python3.10-venv espeak mosquitto mosquitto-dev mosquitto-clients cmake
```

## 2) Create Python Virtual environment and install the needed python modules
Note Conda works well too, and I prefer it if you need a dirrerent python version than the system python3 but Conda  has issues being launched via a shell script via a node-red exec node.  If you know a solution, please send it to me.  I'm not very good with GitHub so you may need to help me with doing "pull requests" if it is not something you can just Email to me.
Now create a virtual environment to to use with OpenVINO and YOLO8, named y8ovv.
```
python3 -m venv y8ovv
```
Next "activate" the virtual environment:
```
source y8ovv/bin/activate
```
Note that the prompt changes from: ai@YouSeeX1: to: (y8ovv) ai@YouSeeX1:
Use pip to install the needed modules:
```
pip install imutils paho-mqtt requests
pip install "openvino>=2024.2.0" "nncf>=2.9.0"
pip install "torch>=2.1" "torchvision>=0.16" "ultralytics==8.2.24" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
```
If not using an Nvidia GPU and Cuda install the Intel GPU driver:
```
sudo apt-get install intel-opencl-icd
```
Add the ai user (your login if you didn't use ai as your username at installation) to the render group:
Make sure the following command is being done as user, not as sudo -i
```
sudo adduser $USER render
```
Now log out and login or reboot the system. GPU doesn't work if you don't do this!
Instructions for using CUDA will be given below, but you will still need OpenVINO for the MobilenetSSD_v2 initial AI if not using the Coral TPU, whos instructions are also given below.

## 3) Clone this repo
Rename the directory to AI2, otherwise you'll need to edit the node-red scripts to account for the different names, then activate the virtual environment:
```
source y8ovv/bin/activate
cd AI2
# make sure the five *.sh scripts have the execute bit set.
chmod ugo+x *.sh
```
Next we need to tell the python code how to talk to the cameras. Two types of cameras are supported, Onvif and RTSP.  Onvif is an overly complited "standard" that is rarely implimented fully or correctly but if you can retrieve an image with an HTTP request it is an Onvif camera for our purposes.  RTSP opens a connection on port 554 and returns a video stream, these are the most common type of cameras.  Be aware that RING, SimplySafe, Arlo, Blink etc. don't generally allow direct access to the video streams or still images.  Also the low end securtiy DVRs like Swann, NightOwl also usually lack support for RTSP streams.  Before you buy, make sure the camera or DVRs you are condidering support RTSP streams and or "Onvif snapshots".

NOTE.  The models are too large to upload to GitHub and for some it may not be allowed.  The Ultralytics yolo8 model is automatically downloaded and converted to openvino format if the files are not there already.  But the MobilnetSSDv2 for the TPU and openvino are not.  You can download the TPU model from:  https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite put it in the AI2/mobilenet_ssd_v2 folder, the coco_labels.txt should already be there.  For the openvino SSD model I converted it using openvino 2021.3 using this guide: https://medium.com/@runeskovrup/object-detection-using-openvino-58b8fe6efbda but it is too old now and has some broken links.  I'm currently looking for a better solution, the 2024 openvino has a lot of model conversions built in, I just need to find the original model source and go up the learning curve to convert it to openvino format for the CPU.


To tell the python code how to use your cameras you need to create a cameraURL.txt file for Onvif cameras or a cameraURL.rtsp file for RTSP cameras.  A cameraURL.txt file should one line per camera containing the HTTP URL to retrieve and image and after a space contain the optional name for what the camera is viewing.
A cameraURL.txt file for two cameras would look like this:
```
http://192.168.2.144:85/images/snapshot.jpg Driveway
http://admin:password@192.168.2.197:80/tmpfs/auto.jpg  Garage
```
Note than some cameras don't require any authorization to return the image, others do and the URLs can be very different format.  It is easy to test your URL simpley by entering into a web browser and you should see and image after connecting and a fresh image after refreshing the page.

A cameraURL.rtsp file should look like this:
```
rtsp://admin:passwd@192.168.2.28:554/cam/realmonitor?channel=1&subtype=0  MailBox
rtsp://admin:passwd%@192.168.2.28:554/cam/realmonitor?channel=7&subtype=0  DriveWay
rtsp://admin:passwd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Garage
rtsp://admin:PassWd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Patio
rtsp://admin:PaSwrd@192.168.2.196:554/h264Preview_01_main FrontYard
rtsp://192.168.2.77:554/stream0?username=admin&password=CE04588A3231F95BEE71848F5958CF4E BackYard
```
The The IP addresses will be what your router assigns, my example shows two security DVRs at xxx.xxx.xxx.28 and xxx.xxx.xxx.49 and two IP or "netcams" the last netcam generates the password as part of the Onvif setup and can be "tricky" to figure out, but they are not generally unique.  Best way I know to test your RTSP URLs is with VLC and "Open Network Connection" and enter the RTSP URL string.  Main negative of RTSP streams is the latency is typically 2-4 seconds behind "real time".

Once the camera URLs are specified (both types of camera files are allowed) we can run a quick test, make sure the virtual environment is active and open up a command window (terminal):
```
python AI2.py -d
```
This will start a thread for each video camaera and an OpenVINO CPU AI thread and display live results on the screen.  Here is an image of it running on a Lenovo IdeaPad i3 laptop doing six Onvif cameras:
![IdeaPad](https://github.com/user-attachments/assets/647f3c54-a265-406a-833d-0ef1a2883eec)  Notifications and the housekeeping functions are done with node-red which we will install next.  Press Ctrl-C in the terminal window to exit the AI python code.

## 4) Install Node-red
Node-red has a very active and helpful community with lots of good tutorials, a vist to their support forum is recommended: https://discourse.nodered.org/

Exit the Python Virtual environment by typing "deactivate" in the terminal where you ran the AI code test.  Install node-red, choose N for Pi specific modules:
```
bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
```
As the script runs It asks you a few questions, these are my answers:
```
would you like to customise the settings now (y/N) ?  Y
=====================================
This tool will help you create a Node-RED settings file.

✔ Settings file · /home/ai/.node-red/settings.js

User Security
=============
✔ Do you want to setup user security? · No

Projects
========
The Projects feature allows you to version control your flow using a local git repository.

✔ Do you want to enable the Projects feature? · Yes
✔ What project workflow do you want to use? · manual - you must manually commit changes

Editor settings
===============
✔ Select a theme for the editor. To use any theme other than "default", you will need to install @node-red-contrib-themes/theme-collection in your Node-RED user directory. · default

✔ Select the text editor component to use in the Node-RED Editor · monaco (default)

Node settings
=============
✔ Allow Function nodes to load external modules? (functionExternalModules) · Yes

```
To complete the node-red setup, in the terminal window do:
```
cd $HOME
node-red-stop
cd ~/.node-red
npm i node-red-dashboard node-red-node-email node-red-contrib-image-output node-red-node-base64
npm install point-in-polygon
```

Now have to make a minor change to the node-red settings file (should still be in the .node-red hidden directory):
```
sudo nano settings.js
```
My "in alert region" node-red filter needs an extra JavaScript module that is not available in node-red by default.
So must edit .node-red/settings.js and add (around lin 510 on new install 30JUL2024), ^/ will let you jump to line 510
and edit it to look like this:
```
      functionGlobalContext: {
          insidePolygon:require('point-in-polygon'),
      }
```
Exit nano with: Ctrl-X Y
restart node red with (remember we stopped it previously):
```
node-red-start
```
And leave the script by typing Ctrl-C in the termainal after the node-red "start-up" messages, then make node-red start automatically with:
```
sudo systemctl enable nodered.service
```

If you didn't do this is step 0), do it now to avoid having to edit all the scripts used by node-red exec nodes in the sample controller flow:
```
sudo ln -s /home/YourUserName /home/ai
```

To install the basic controller, open web browser (Chromium is recommended) and point it at YourHostName:1880 (or localhost:1880 if not installing remotely) and follow through the "Welcome to Node-RED 4.0" steps to see what is new and different from prior versions. When you get the "projects" do "Create Project" and fill in the dialogs.  I chose NO to security and encryption since no external connections are accepted by my firewall, do what works for you.

Open the node-red "Hamburger" (three parallel bars) menu in the upper right corner of the editor and select "Import".  Press the "new flow" buton in the dialog box click the "select a file to import" button.  Navigate to the BasicAI2Controller.json. file.  Afterwards, click the Deploy button and you should be ready to go.  you'll have to set up the Email addresses in the "Setup Email" node and set up your smtp Email account credentials in the "Email Notification" node.  You will also have to edit the AI2/StartAI.sh file to use the correct options to start the AI2 Python code, hopefully the comments in the file help guide you.


## 5) Setup to use the Coral TPU
Add the "current" coral repo:
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```
If you want TPU to run full speed:
```
sudo apt install libedgetpu1-max
```

If you want to use the M.2, mPCIe, etc. TPUs:
Depending on when and where you install you can get different kernel versions.
Virgin install on 1AUG2024 on Celeron got: 6.5.0-45-generic #45~22.04.1-Ubuntu
on my i9 setup in 2022 I got: 5.15.0-113-generic #123-Ubuntu
Don't install this if you have v6 kernel but it is not fatal, can "sudo apt remove" it later if necessary.
```
sudo apt install gasket-dkms
```
NOTE: the dkms build fails on kernel v6.+ what worked to fix it:
```
sudo apt remove gasket-dkms     # only if you did apt install gasket-dms on v6+ kernel
sudo apt install devscripts debhelper -y
git clone https://github.com/google/gasket-driver.git
cd gasket-driver; debuild -us -uc -tc -b; cd ..
sudo dpkg -i gasket-dkms_1.0-18_all.deb
```
Next setup the apex user and verify the PCI-E installation:
```
sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"
sudo groupadd apex
#!!! Make sure the following command is being done as ai user, not as sudo -i
sudo adduser $USER apex
```
Now reboot the system.
Once rebooted, verify that the accelerator module is detected:
```
lspci -nn | grep 089a
```
You should see something like this:
03:00.0 System peripheral: Device 1ac1:089a
The 03 number and System peripheral name might be different, because those are host-system specific, 
but as long as you see a device listed with 089a then you're okay to proceed.

Also verify that the PCIe driver is loaded:
```
ls /dev/apex_0
```
You should simply see the name repeated back:
/dev/apex_0

## 6)  Setup CUDA, I'm no expert, but this is what I did.
I think it is safest to use the Ubuntu repo version of the nvidia driver and install cuda from downloaded Nvidia deb file.
Install the cuda toolkit:
```
sudo apt install nvidia-cuda-toolkit
```
Download from the NVidia site: cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
Note that this depends on the Nvidia driver you've installed, I used 510.108.03 which is not the "latest", but when I did
it on a "virgin" system it seemed to installed driver 515.105.01, which required a reboot for nvidia-smi to work
```
sudo dpkg -i Downloads/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-F83D2C4C-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt-get -y install cuda
```
Next install the CUDNN:
Download: cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz from Nvidia site,
Extract the archive from the file manager and copy the include and lib directories to /usr/local
```
cd Downloads/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive
sudo mkdir /usr/local/cuda-11.7
sudo cp -a include /usr/local/cuda-11.7
sudo cp -a lib /usr/local/cuda-11.7
```
CUDA is like OpenVINO, quite dynamic, so my notes here from about two years ago may not be the best bet now,  I'll happily accept updates to these instructions!
To launch my AI 

## 7) Some useful options:
#### It is best to put AI detections on a seperate drive or USB stick, but not necessary,
I had to mount the external device to create the /media/ai directory where it will mount,
then unmount it and do:
```
sudo mkdir /media/ai
sudo mkdir /media/ai/AI
sudo chown ai.ai /media/ai
sudo chown ai.ai /media/ai/AI
```
#### then edit /etc/fstab so it mounts on bootup:
```
sudo nano /etc/fstab
```
Add entry like this, changing /dev and mount point as desired (here AI is the drive's ext4 label name)
Note tgat /dev/sda1 is usually the name of a USB stick for Linux system on eEMC, SD card or NVMe drive, but not always
Hint, use the Linux "Disks" utility programe to get the device name of the partition to mount.
```
# My YouSeeToo-X1 system is booting from eEMC and using NVMe drive to store images:
/dev/nvme0n1p1 /media/ai/AI	ext4	defaults,nofail,noatime	0	3
# For a USB stick
/dev/sdb1	/media/ai/AI	ext4	defaults,nofail,noatime	0	3
```
Then make a symlink in /home/ai:
```
cd $HOME
ln -s /media/ai/AI detect
```
Reboot and verify the external drive is mounted.

#### I also like to setup a samba server so you can view the detection images from other machines on your local subnet.
Configue samba file sharing.  Since this is to be an "appliance" this is the best way to look at AI detection files, edit samba config:
```
sudo nano /etc/samba/smb.conf
```
Add this is in the [global] section:
```
    mangled names = no
    follow symlinks = yes
    wide links = yes
    unix extensions = no
    # Ubuntu 20.04 seems to require these:
    server min protocol = NT1
    client min protocol = NT1
```
Make the homes section be like this:
```
[homes]
   comment = Home Directories
   browseable = yes
   read only = no
   writeable = yes
   create mask = 0775
   directory mask = 0775
```
#### set samba password, I use the ai login password for simplicity:
```
sudo smbpasswd -a ai
```

