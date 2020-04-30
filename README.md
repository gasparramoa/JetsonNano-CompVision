# JetsonNano-CompVision
Simple guideline for installing all the main computer vision libraries in the SBC Jetson Nano

https://we.tl/t-RZW0o5nfdX

## Recommended Requesites in JetsonNano
At least a 32 GB SD-Card (I'm using a 64 GB)\
USB fan to reduce CPU Tempeature to avoid CPU throttle.


### Increase Swap memory (4 Gb at least recommended)
```
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Dependencies of deep learning frameworks and libraries
```
sudo apt install -y git
sudo apt install -y cmake
sudo apt install -y libatlas-base-dev
sudo apt install -y gfortran
sudo apt install -y python3-dev
sudo apt install -y python3-pip
sudo apt install -y libhdf5-serial-dev
sudo apt install -y hdf5-tools 
```

### Install mlocate library for (updatedb and locate) work
``` sudo apt install mlocate ```

### Install setuptools numpy and matplotlib
```
pip3 install numpy (NOT WORKING)
pip3 install -U pip setuptools
pip3 install matplotlib (NOT WORKING)
```

### Realsense tools for 3D Realsense Camera
```
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u\
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg 
```

Test it by typping: 
```realsense-viewer```

### python-dev and htop (optional)
```
sudo pip3 install python-dev-tools 
sudo apt-get install python3.6-devel 
sudo apt update && sudo apt upgrade 
sudo apt install htop
```

### Install OpenCV
```
wget https://raw.githubusercontent.com/AastaNV/JEP/master/script/install_opencv4.1.1_Jetson.sh 
mkdir opencv 
git clone https://github.com/AastaNV/JEP.git 
cd JEP/script/ 
./install_opencv4.1.1_Jetson.sh ~/opencv  [Takes some time (2h-4h), depends of power source and cooler] 
```
Add to the ```.bashrc``` file the following: \
``` export PYTHONPATH=$PYTHONPATH:/<where-you-place-jep-folder>/JEP/script/opencv-4.1.1/release/python_loader/ ``` \
``` import opencv``` should work in python

### Install Open3D (Manipulate 3D Data)
```
git clone https://github.com/intel-isl/Open3D.git 
util/scripts/install-deps-ubuntu.sh 
mkdir build 
cd build 
cmake .. 
sudo make -j4 
sudo make install-pip-package (wheel)
```
``` import open3d``` should work in python


### Install Python wrapper for realsense (pyrealsense2)
```
sudo apt-get dist-upgrade
sudo apt-get install python3 python3-dev 
git clone https://github.com/IntelRealSense/librealsense.git 
cd librealsense 
mkdir build 
cd build 
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true 
make -j4 
sudo make install 
export PYTHONPATH=$PYTHONPATH:/usr/local/lib (or place it in the ./bashrc file)
```
``` import pyrealsense2``` should work in python

### Install keyboard library
```sudo pip3 install keyboard``` \
```import keyboard``` should work in python


### Install PyThorch(Version 1.0.0) and Torchvision
```
wget https://nvidia.box.com/shared/static/2ls48wc6h0kp1e58fjk21zast96lpt70.whl -O torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl 
sudo pip3 install numpy torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl

cd /usr/local/lib/python3.6/site-packages or /<virtual-env-dir>/lib/python3.6/site-packages
git clone https://github.com/pytorch/vision
cd vision
sudo python3 setup.py install
```
or
```
sudo pip3 install torchvision
```

add to the ```bashrc file``` the following:
``` 
export CUDADIR=/usr/local/cuda-10.0
export PATH=$PATH:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
```

### Install Terminator (Terminal with less consuming memory - RAM)
```
sudo apt install terminator
```

### Install Context Encoding (optional) for Semantic Segmentation (PyTorch) - NOT WORKING
```
sudo apt-get install ninja-build
python3 setup.py install
```
Add the following lines to the ``` .bashrc ```file \
My ninja directory is ``` \usr\bin\ninja ``` and cuda directory is ``` \usr\local\cuda-10.0 ``` but it can depend.
```
export PATH=<your-ninja-directory>:${PATH}
export CUDA_HOME=<your-cuda-directory>
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

### Install PointNet (optional) for 3D Object Classification (PyTorch)
```
git clone https://github.com/fxia22/pointnet.pytorch.git
cd pointnet.pytorch
pip3 install -e .
cd scripts
bash build.sh
bash download.sh 
```

### Install other usefull libraries:
``` 
sudo apt-get install python3-matplotlib 
sudo pip3 install scipy
```

### Install Fully Convolutional (optional) HarDNet for Segmentation (Pytorch)
Dependencies:
``` 
pip3 install pydensecrf
pip3 install protobuf
pip3 install tensorboardX
pip3 install imageio
```
Download data for desired dataset. \
Download repository: ```git clone https://github.com/PingoLH/FCHarDNet``` \
Create setup file: ```gedit config.yaml``` \
Extract the zip and modify the path in the ```config.yaml``` file. \
To train the model:
```
python3 train.py --config [config.yaml]
```

### Install i3 (optional) - lightweight graphic interface (Faster)
``` 
sudo apt install i3 
sudo reboot
```
Select option in the login screen - ``` i3 ``` \
Shortcuts:
```
windowns + Shift + q -> close selected application
windowns + Enter -> open new terminal
windowns + d -> open search menu
windowns + (number of workspace 0-9) -> change workspace 
windowns + shift + (nº of workspace) -> move/send to workspace
windowns + f -> fullscreen current aplication
windowns + r -> resize selected windown -> Use arrows to resize and "Esc" to leave the resize mode
```
Configuration file to create new shortcuts:
```
home/<your-user>/.config/i3/config
```    

## Speed up the Start up in Jetson Nano

### remove uncessary startup programs

### enable auto login in i3 ubuntu
```
sudo gedit /etc/gdm3/custom.conf
```
uncomment the two lines below ``` # Enabling automatic login ```


## Start python-script after boot
Add the following to the ``` ~/.bashrc ``` file:
```
echo Running python script
python3 your_python_script.py
```
What is left is to open the terminal after boot to run the script.
```
cd ~/.config/i3
```
add the following to the ```config``` file:
```
for_window [class="Terminal"] move container to workspace 1
exec --no-startup-id i3-sensible-terminal
```



# Configuration - Computer for training (PoP!_OS 19.10)

## Install DIGITS for jetson-inference
The installation of DIGITS must be made by source because the POP!_OS_! system is incompatible with the nvidia docker.
Follow these instructions: https://github.com/NVIDIA/DIGITS/blob/digits-6.0/docs/BuildDigits.md \
The command ``` ./digits-devserver ``` may not work, because of the flask_socketio package. \
Remove the flask_socketio package from the requeriments.txt inside digits folder. \
Install the last version by doing: ``` pip install flask_socketio ``` .\
Them you might have another error, to solve upgrade the werkzeug library, ``` pip install --upgrade werkzeug==0.16.1 ``` \
\
To start the server simply use the command: ``` ./digits-devserver ``` \

In case of error in detection models, just do: ``` sudo pip install --user --upgrade protobuf==3.1.0.post1 ```


# Jetpack 4.3 to Jetpack 4.4

### Save SD-Card Jetpack 4.3

First use command ``` df -h ``` to see the available devices. \
Plug the SDCard in your computer and use the command ``` df -h ``` again. \
Now you should see a new line and that's your SDCard, normally it looks like ``` /dev/mmcblk0p1 ```\
Use ``` sudo dd if=/dev/mmcblk0 of=~/SDCardBackupJetpack43.img ``` \
This previous command will save the current image (jetpack4.3) of your SD-Card in ``` /SDCardBackupJetpack43.img ``` \
It will take some time (around 20~30 minutes).

### Format SD-Card
For formating the SDCard use ``` gparted ``` and remove all the partitions and format in ``` fat32 ```

### Download Jetpack4.4 image
What's left to do is to download the image from: https://developer.nvidia.com/embedded/jetpack-archive\
And follow the Jetson Nano instructions: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write 

