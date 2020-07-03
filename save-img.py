# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np
import subprocess
import os
import cv2
import time
import keyboard  # using module keyboard
import sys


from datetime import datetime
now = datetime.now() 


# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
#Start streaming with default recommended configuration

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

cfg = pipe.start(config);


#profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
#intr = profile.as_video_stream_profile().get_extrinsics() # Downcast to video_stream_profile and fetch intrinsics
#   print(intr)

#sys.exit()

align_to = rs.stream.color
align = rs.align(align_to)
file_save="./Imagens"


def save_RGB_and_Depth():

    while True:
        #print("Waiting for 'enter'...")

        #a = raw_input("\npress enter:")
        
        #print("time_s=str(int(time.time()))")
        time_s=str(int(time.time()))
            
        #print("2")
        frames = pipe.wait_for_frames()
        

        # Align the depth frame to color frame
        #print("3")
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        #print("4")
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        # aligned_depth_frame is a 640x480 depth image
        #print("5")
        color_frame = aligned_frames.get_color_frame()
        #print("6")
        if not aligned_depth_frame or not color_frame:
            continue

        #print("7")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        #print("8")
        now = datetime.now() 

        print(depth_image)
        
        


        #print("after enter")

        # Quando receber o sinal, ira guardar RGB + Depth

        '''pasta = str(now.month)
        
        os.mkdir("Imagens/" + pasta)
        os.chmod("Imagens/" + pasta,0o777) # Para ter premissões depois de criar a pasta... '''


        #print("saving..")
        #print(now)

        cv2.imwrite("frame_"+str(now)+".png",color_image)
        cv2.imwrite("depth_"+str(now)+".png",depth_image)
        #print("/home/pi/librealsense/build/wrappers/python/Imagens/"+a+"/frame_"+str(now)+".png")
        #print("/home/pi/librealsense/build/wrappers/python/Imagens/"+a+"/depth_"+str(now)+".png")
        #sys.exit()

try:
    print("Begin...")
    save_RGB_and_Depth()
finally:
    pipe.stop()

