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


from gtts import gTTS
from playsound import playsound

import multiprocessing
from statistics import mean 

def speak(text):

    language = "en"  # pt # language
    file_name = text + ".mp3"  # name of where the file will be saved

    output = gTTS(text=text, lang=language, slow=False)  # configure tts
    output.save(file_name)  # create file

    playsound(file_name)

#os.system("play " + ficheiro)





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
    #classifier.load_state_dict(torch.load('/home/socialab/FCHarDNet/cls_model_40.pth'))
    classifier.load_state_dict(torch.load('/home/socialab/human_vision/FCHarDNet/cls_model_40.pth'))    
    classifier.eval()
    end = time.time()
    print("  (time): " + str(end-start))

def filtrar_segmentacao(imagemDepth2):


    start = time.time()


    imageDepth2 = imagemDepth2


    a_imageDEPTH2 = np.asarray(imageDepth2)

    o3d_Depth2 = o3d.geometry.Image(a_imageDEPTH2)

    pcd2 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_Depth2,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

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
        playsound('DoorClosed.mp3')
    else:
        print("\n [OPEN DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")
        playsound('DoorOpen.mp3')

def run_first_mode():
    os.system("python3 Street_mod_v1.py")

def percorre_matriz_imagem2(procc_number, matriz, linha_i, linha_f, freq_bip):
    
    freq_sound = 0
    # freq_sound -> 0, 1, 2, 3
    ele_especial = -1

    flag = 0

    medias = []

    for i in range(linha_i, linha_f):
        medias.append(mean(matriz[i]))

    meean = mean(medias)
    freq_bip[procc_number] = meean
    print("medias: " + str(meean))






def main_program():


    # Initial sound of the program

    duration = 0.2  # seconds
    freq = 500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 400))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, 600))

    inicializar_pointnet()
    


    change_mode = 0

    #for i in range(12):

    while True:


        if change_mode == 5:
                        
            print("Change mode")

            os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 700))
            os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 450))
            os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 340))
            
            manager = multiprocessing.Manager()
            p = multiprocessing.Process(target=run_first_mode)
            p.start()

            sys.exit()




        program_time = time.time()

        frames = pipe.wait_for_frames()

        threshold_depth = threshold.process(frames.get_depth_frame())

        depth_image2 = np.asanyarray(threshold_depth.get_data())

        image_depth_2 =Image.fromarray(depth_image2)

        image_depth_22 = image_depth_2.rotate(270, expand=True)   

        end = time.time()
        print("Realsense time (Get depth frames, rotate image)\n: " + str(end-program_time))      

        filtrar_segmentacao(image_depth_22)

        program_time_end = time.time()
        total_program_time = program_time_end-program_time
        print("Total time Program: "+str(program_time_end-program_time) + " sec\n\n")

        processes = []
        i_linha = 0
        f_linha = 60

        manager = multiprocessing.Manager()
        freq_bips = manager.dict()

        for i in range(8):
            p = multiprocessing.Process(target=percorre_matriz_imagem2, args=(i, depth_image2, i_linha, f_linha, freq_bips))
            p.start()
            processes.append(p)
            i_linha = i_linha + 60
            f_linha = f_linha + 60

        freq_bips_array = freq_bips.values()

        if mean(freq_bips_array) < 50:
            change_mode = change_mode + 1
        else:
            change_mode = 0

        print("change_mode: " + str(change_mode))

try:
    main_program()
finally:
    pipe.stop()

