# -*- coding: utf-8 -*-

#####################################################
##      Door Problem - PointNet and FCHarDNet      ##
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

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict


# Declare RealSense pipeline, encapsulating the actual device and sensors.
pipe = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

pipe.start(config);
align_to = rs.stream.color
align = rs.align(align_to)

# Global Variables:
model = ""
classifier = ""
imagemSegmentacao = ""
loader = ""

#Semantic Segmentation



def avaliar_segmentacao(imagemRGB):

    global imagemSegmentacao


    resized_img = np.resize(imagemRGB, (loader.img_size[0], loader.img_size[1], 3))
    orig_size = imagemRGB.shape[:-1]

    img = imagemRGB

    img = np.resize(img, (loader.img_size[0], loader.img_size[1], 3))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)


    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0) 
    pred = np.resize(pred, (orig_size[0], orig_size[1]))


    decoded = loader.decode_segmap(pred)

    decoded = np.array(decoded, dtype=np.uint8)

    imagemSegmentacao = decoded


def inicializar_segsem():

    print("Loading Semantic Segmentation Model:")
    start = time.time()
    global loader
    global device
    global model

    device = torch.device("cuda")

    model_name = "hardnet"
    data_loader = get_loader("ade20k")
    loader = data_loader(root=None, is_transform=True, img_norm=True, test_mode=True)
    n_classes = loader.n_classes

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version="ade20k")
    state = convert_state_dict(torch.load("/home/socialab/FCHarDNet/runs/config./cur/hardnet_ade20k_best_model.pkl",)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    end = time.time()
    print("  (time): " + str(end-start))


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

def filtrar_segmentacao(imagemRGB, imagemDepth):

    global imagemSegmentacao

    start = time.time()
    trans = transform.ToPILImage() 
    trans1 = transform.ToTensor()

    print(imagemSegmentacao.dtype)

    imagemSegmentacao_IMAGE = trans(imagemSegmentacao)
    image_antes= imagemSegmentacao_IMAGE.convert('RGB')
    image = np.asarray(image_antes)

    imageRGB = imagemRGB
    imageDepth = imagemDepth

    a_imageRGB = np.asarray(imageRGB)
    a_imageDEPTH = np.asarray(imageDepth)

    #lower = np.array([30,80,130])
    #upper = np.array([30,80,130])

    lower = np.array([0,0,0])
    upper = np.array([0,0,0])

    mask = cv2.inRange(image, lower, upper)

    output = cv2.bitwise_and(image, image, mask = mask)

    ret,thresh = cv2.threshold(mask, lower[0], 255, 0)

    se = np.ones((50,50), dtype='uint8')
    image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

    contours,hierarchy = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        height, width, channels = image.shape

        if (w * h >= 30000 and (w >= 150)):
            limit_bouding = 5
            x_save = 1
            y_save = 1
            xf_save = width
            yf_save = height

            if x > limit_bouding:
                x_save = x - limit_bouding                 
            if y > limit_bouding:
                y_save = y - limit_bouding
            if (height - (y+h)) > limit_bouding:
                yf_save = y+h+limit_bouding
            if (width - (x+w)) > limit_bouding:
                xf_save = x+w+limit_bouding

            # Try to save more 5 pixels than bouding box.

            img_RGB_crop = o3d.geometry.Image((a_imageRGB[y_save: yf_save, x_save: xf_save]).astype(np.uint8))
            img_Depth_crop = o3d.geometry.Image((a_imageDEPTH[y_save: yf_save, x_save: xf_save]).astype(np.uint16))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_RGB_crop, img_Depth_crop)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #o3d.visualization.draw_geometries([pcd])

            #Convert to point-net format:
            pcd_load = pcd
            xyz_load = np.asarray(pcd_load.points) 
            
            end = time.time()
            print("[Filtering Image] time: " + str(end-start))

            start = time.time()
            pointnet(xyz_load)
            end = time.time()
            print("[PointNet] time: "+str(end - start))   

            
        else:
            print("Door wasn't detected [contours too small]")  
    else:
        print("Door wasn't detected [Couldn't find any countours...]")

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

    inicializar_segsem()

    inicializar_pointnet()
    
    while True:
        a = input("\n ** Predict [Open/Closed] (ENTER):")

        program_time = time.time()
        
        frames = pipe.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Images RGB and Depth
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        image_color = Image.fromarray(color_image)
        image_depth = Image.fromarray(depth_image)

        image_color2 = image_color.rotate(270, expand=True)
        image_depth2 = image_depth.rotate(270, expand=True)

        image_color2 = np.asarray(image_color2)

        #FAZER SEGMENTAÇÃO DA MESMA
        start = time.time()
        avaliar_segmentacao(image_color2)
        end = time.time()
        print("[Semantic Segmentation] time: "+str(end - start))        

        #FILTRAR IMAGEM COM SEGMENTAÇÃO.
        filtrar_segmentacao(image_color2, image_depth2)

        program_time_end = time.time()
        print("Total time Program: "+str(program_time_end-program_time) + " sec")



try:
    main_program()
finally:
    pipe.stop()

