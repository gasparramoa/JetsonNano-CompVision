# JetsonNano-CompVision
Simple guideline for installing all the main computer vision libraries in the SBC Jetson Nano


## Recommended Requesites in JetsonNano
At least a 32 Gb SD-Card \
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
pip3 install numpy
pip3 install -U pip setuptools
pip3 install matplotlib
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
bash install_opencv4.0.0_Nano.sh $HOME/.local 
wget https://raw.githubusercontent.com/AastaNV/JEP/master/script/install_opencv4.0.0_Nano.sh 
mkdir opencv 
git clone https://github.com/AastaNV/JEP.git 
cd JEP/script/ 
./install_opencv4.1.1_Jetson.sh ~/opencv  [Takes some time (4h)] 
```
Add to the ```.bashrc``` file the following: \
``` export PYTHONPATH=$PYTHONPATH:/<where-you-place-jep-folder>/JEP/script/opencv-4.1.1/release/python_loader/ ```

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

### Install keyboard library
```sudo pip3 install keyboard```

### Install PyThorch(Version 1.0.0) and Torchvision
```
wget https://nvidia.box.com/shared/static/2ls48wc6h0kp1e58fjk21zast96lpt70.whl -O torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl 
sudo pip3 install numpy torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl

cd /usr/local/lib/python3.6/site-packages or /<virtual-env-dir>/lib/python3.6/site-packages
git clone https://github.com/pytorch/vision
cd vision
sudo python3 setup.py install
cd ~
```

### Install Terminator (Terminal with less consuming memory - RAM)
```
sudo apt install terminator
```

### Install Context Encoding (optional) for Semantic Segmentation (PyTorch)
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
cd script
bash build.sh
bash download.sh 
```

### Install Matplotlib
``` sudo apt-get install python3-matplotlib ```

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
```
Configuration file to create new shortcuts:
```
home/<your-user>/.config/i3/config
```    
