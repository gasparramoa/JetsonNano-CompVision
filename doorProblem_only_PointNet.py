# -*- coding: utf-8 -*-

#####################################################
##          Door Problem - PointNet Only           ##
#####################################################


import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import subprocess
import os
import cv2
import time
import keyboard  
import sys
import torchvision.transforms as transform
import torch
import imageio



from PIL import Image
from tqdm import tqdm
from torch.utils import data

from datetime import datetime
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
from torch.autograd import Variable
from matplotlib import pyplot as plt




# Declare RealSense pipeline, encapsulating the actual device and sensors.
pipe = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe.start(config);
align_to = rs.stream.color
align = rs.align(align_to)
threshold = rs.threshold_filter()
threshold.set_option(rs.option.max_distance, 5)


# Global Variables:
model = ""
classifier = ""
imagemSegmentacao = ""
loader = ""

def inicializar_pointnet():
    
    print("Loading 3D Object Classification Model:")
    start = time.time()
    global classifier
    classifier = PointNetCls(k=2) #IMP, 2 é o número de classes, depois alterar.
    classifier.cuda()
    classifier.load_state_dict(torch.load('/home/socialab/FCHarDNet/cls_model_40.pth'))
    classifier.eval()
    end = time.time()
    print("  (time): " + str(end-start))

def filtrar_segmentacao(imagemDepth2):


    start = time.time()


    #imageRGB = imagemRGB
    #imageDepth = imagemDepth
    imageDepth2 = imagemDepth2

    #a_imageRGB = np.asarray(imageRGB)
    #a_imageDEPTH = np.asarray(imageDepth)
    a_imageDEPTH2 = np.asarray(imageDepth2)

    #print(a_imageDEPTH2)

    #o3d_RGB = o3d.geometry.Image(a_imageRGB)
    #o3d_Depth = o3d.geometry.Image(a_imageDEPTH)
    o3d_Depth2 = o3d.geometry.Image(a_imageDEPTH2)

    #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_RGB, o3d_Depth)

    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #rgbd_image,
        #o3d.camera.PinholeCameraIntrinsic(
            #o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


    pcd2 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_Depth2,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #pcd2=pcd2.voxel_down_sample(voxel_size=0.00001)

    # Flip it, otherwise the pointcloud will be upside down
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #o3d.visualization.draw_geometries([pcd])

    #o3d.visualization.draw_geometries([pcd2 + pcd])
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd2])

    #Convert to point-net format:
    #pcd_load = pcd
    #xyz_load = np.asarray(pcd_load.points)

    #print(xyz_load.shape)
    #print(np.asarray(pcd2.points).shape) 
    
    end = time.time()
    print("[Depth to pointcloud time]: " + str(end-start))

    start = time.time()
    pointnet(np.asarray(pcd2.points))
    end = time.time()
    print("[PointNet] time: "+str(end - start))   


def pointnet(points_sett):
    #batch_size = BS
    BS = 1

    point_set = points_sett
    point_set = np.float32(point_set)

    choice = np.random.choice(len(point_set), 10000, replace=True)
    #resample
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale

    points_sets =np.array([point_set])

    points_sets = torch.from_numpy(points_sets)

    points = points_sets
    points = Variable(points)
    points = points.transpose(2, 1)
    points = points.cuda()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]

    if str(pred_choice).split("tensor([")[1].split("]")[0] == '0':
        print("\n [CLOSED DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")
    else:
        print("\n [OPEN DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")



def main_program():


    inicializar_pointnet()
    
    #for i in range(12):

    while True:
        a = input("\n[>> Predict [Open/Closed] (ENTER):")

        program_time = time.time()

        frames = pipe.wait_for_frames()
        
        # Big problem in using depth and RGB information
        # Align the depth frame to color frame
        #aligned_frames = align.process(frames)
      
        #ThresHOLD!!
        threshold_depth = threshold.process(frames.get_depth_frame())

        # Get aligned frames
        #aligned_depth_frame = aligned_frames.get_depth_frame() 
        # aligned_depth_frame is a 640x480 depth image
        #color_frame = aligned_frames.get_color_frame()

        #if not aligned_depth_frame or not color_frame:
            #continue


        # Images RGB and Depth
        #depth_image = np.asanyarray(aligned_depth_frame.get_data())
        #color_image = np.asanyarray(color_frame.get_data())
        depth_image2 = np.asanyarray(threshold_depth.get_data())

        #depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


        #image_color = Image.fromarray(color_image)
        #image_depth = Image.fromarray(depth_image)
        image_depth_2 =Image.fromarray(depth_image2)

        #image_color2 = image_color.rotate(270, expand=True)
        #image_depth2 = image_depth.rotate(270, expand=True)
        image_depth_22 = image_depth_2.rotate(270, expand=True)   

        #image_color2 = np.asarray(image_color2)

        end = time.time()
        print("Realsense time (Get depth frames, rotate image)\n: " + str(end-program_time))      


        #FILTRAR IMAGEM COM SEGMENTAÇÃO.
        #filtrar_segmentacao(image_color2, image_depth2, image_depth_22)
        filtrar_segmentacao(image_depth_22)

        program_time_end = time.time()
        total_program_time = program_time_end-program_time
        print("Total time Program: "+str(program_time_end-program_time) + " sec\n\n")



try:
    main_program()
finally:
    pipe.stop()

