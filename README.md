# JetsonNano-CompVision
Simple guideline for installing all the main computer vision libraries in the SBC Jetson Nano


## Recommended Requesites in JetsonNano
At least a 32 Gb SD-Card \
USB fan to reduce CPU Tempeature to avoid CPU throttle.


### Increase Swap memory (4 Gb)
sudo fallocate -l 4G /swapfile\
sudo chmod 600 /swapfile\
sudo mkswap /swapfile\
sudo swapon /swapfile\
sudo swapon --show\
sudo cp /etc/fstab /etc/fstab.bak\
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

### Dependencies of deep learning frameworks and libraries
sudo apt install -y git \
sudo apt install -y cmake \
sudo apt install -y libatlas-base-dev \
sudo apt install -y gfortran \
sudo apt install -y python3-dev \
sudo apt install -y python3-pip \
sudo apt install -y libhdf5-serial-dev \
sudo apt install -y hdf5-tools 

### Install mlocate library for (updatedb and locate) work
sudo apt install mlocate

### Install setuptools numpy and matplotlib
pip3 install numpy\
pip3 install -U pip setuptools\
pip3 install matplotlib

### Realsense tools for 3D Realsense Camera
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE\
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u\
sudo apt-get update\
sudo apt-get install librealsense2-dkms\
sudo apt-get install librealsense2-utils \
sudo apt-get install librealsense2-dev \
sudo apt-get install librealsense2-dbg 

### Test Realsense viewer
realsense-viewer 

### python-dev and htop (optional)
sudo pip3 install python-dev-tools \
sudo apt-get install python3.6-devel \
sudo apt update && sudo apt upgrade \
sudo apt install htop

### Install OpenCV
bash install_opencv4.0.0_Nano.sh $HOME/.local \
wget https://raw.githubusercontent.com/AastaNV/JEP/master/script/install_opencv4.0.0_Nano.sh \
mkdir opencv \
git clone https://github.com/AastaNV/JEP.git \
cd JEP/script/ \
./install_opencv4.1.1_Jetson.sh ~/opencv  [Takes some time (4h)] 

### Install Open3D (Manipulate 3D Data)
git clone https://github.com/intel-isl/Open3D.git \
util/scripts/install-deps-ubuntu.sh \
mkdir build \
cd build \
cmake .. \
sudo make -j4 \
sudo make install-pip-package (wheel)

### Install Python wrapper for realsense (pyrealsense2)
sudo apt-get dist-upgrade \
sudo apt-get install python3 python3-dev \
git clone https://github.com/IntelRealSense/librealsense.git \
cd librealsense \
mkdir build \
cd build \
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true \
make -j4 \
sudo make install \
export PYTHONPATH=$PYTHONPATH:/usr/local/lib

### Install keyboard library
sudo pip3 install keyboard

### Install PyThorch(Version 1.0.0) and Torchvision
wget https://nvidia.box.com/shared/static/2ls48wc6h0kp1e58fjk21zast96lpt70.whl -O torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl \
sudo pip3 install numpy torch-1.0.0a0+bb15580-cp36-cp36m-linux_aarch64.whl 

### Install Context Encoding (optinal) for Semantic Segmentation
git clone https://github.com/zhanghang1989/PyTorch-Encoding.git \
python3 setup.py install
