# -*- coding: utf-8 -*-

#####################################################
##              Street        -      Mod           ##
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
import sys
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
import simpleaudio as sa
import wave


# Declare RealSense pipeline, encapsulating the actual device and sensors.
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe = rs.pipeline()

pipe.start(config);
align_to = rs.stream.color
align = rs.align(align_to)
threshold = rs.threshold_filter()
threshold.set_option(rs.option.max_distance, 10)
#_________________________________________________________________________



# Stereo Audio:
# get timesteps for each sample, T is the note duration in seconds
sample_rate = 44100
T = 0.2272753623188406
valor = sample_rate * T
t = np.linspace(0, T, int(valor), False)



def stereo_beep(left, right):

    if left == 0 and right == 0:
        print("divison 0")

    else:

        wave_read = wave.open("/home/socialab/JetsonNano-CompVision/sounds/beep1_stereo.wav", 'rb')
        audio_data = wave_read.readframes(wave_read.getnframes())

        data = np.fromstring(audio_data, dtype=np.uint16)
        

        data[0::2] = data[0::2] / right # atenção com a divisão por 0, depois resolver.
        data[1::2] = data[1::2] / left


        num_channels = wave_read.getnchannels()
        bytes_per_sample = wave_read.getsampwidth()
        sample_rate = wave_read.getframerate()

        listt = [left, right]
        med = min(listt)
        




        play_obj = sa.play_buffer(data.tostring(), 2, bytes_per_sample, sample_rate)
        play_obj.wait_done()

        if med > 100:
            time.sleep(0.1)
        elif med > 300:
            time.sleep(0.15)

def percorre_matriz_imagem2(procc_number, matriz, linha_i, linha_f, freq_bip):
    
    freq_sound = 0
    # freq_sound -> 0, 1, 2, 3
    ele_especial = -1

    flag = 0

    medias = []

    for i in range(linha_i, linha_f):
        medias.append(mean(matriz[i]))
        '''
        for ele in matriz[i]:
            if ele == 0:
                continue
            if ele <= 300:
                flag = 1
            if ele <= 180:
                flag = 2
            if ele <= 160:
                flag = 3
            
            if flag > freq_sound:
                freq_sound = flag
        '''


    #print("freq_sound: " + str(freq_sound))
    meean = mean(medias)
    freq_bip[procc_number] = meean
    print("medias: " + str(meean))
    #freq_bip[procc_number] = freq_sound






def main_program():

    # Initial sound of the program
    duration = 0.2  # seconds
    freq = 500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, 440))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 400))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, 640))

    change_mod = 0


    while True:

        if change_mod == 5:

            print("Turn Off Program")

            os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 700))
            os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 450))
            os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 340))

            sys.exit()


        program_time = time.time()

        frames = pipe.wait_for_frames()
        
        threshold_depth = threshold.process(frames.get_depth_frame())
        depth_image2 = np.asanyarray(threshold_depth.get_data())

        start = time.perf_counter()


        processes = []
        i_linha = 0
        f_linha = 60

        manager = multiprocessing.Manager()
        freq_bips = manager.dict()

        # 8 Threads, each one with 60 lines of the depth information of the 3D Camera.
        # 480 = 8 * 60
        for i in range(8):
            p = multiprocessing.Process(target=percorre_matriz_imagem2, args=(i, depth_image2, i_linha, f_linha, freq_bips))
            p.start()
            processes.append(p)
            i_linha = i_linha + 60
            f_linha = f_linha + 60

        for process in processes:
            process.join()

        to_sound = mean(freq_bips.values())

        freq_bips_array = freq_bips.values()

        esquerda = []
        esquerda.extend((freq_bips_array[0], freq_bips_array[1], freq_bips_array[2], freq_bips_array[3]))
        m_esquerda = mean(esquerda)

        direita = []
        direita.extend((freq_bips_array[4], freq_bips_array[5], freq_bips_array[6], freq_bips_array[7]))
        m_direita = mean(direita)

        # Change this values to increase or decrease sound volume, example -> 10-volumeHigh,  800-volumeLow

        if m_direita < 50:
            sound_direita = 10
        elif m_direita < 180:
            sound_direita = 300
        elif m_direita < 300:
            sound_direita = 400
        elif m_direita < 500:
            sound_direita = 750
        else:
            sound_direita = 0

        if m_esquerda < 50:
            sound_esquerda = 10
        elif m_esquerda < 180:
            sound_esquerda = 300
        elif m_esquerda < 300:
            sound_esquerda = 400
        elif m_esquerda < 500:
            sound_esquerda = 750
        else:
            sound_esquerda = 0  

        # If the user place his hand in front of the camera during 5 frames, the system will shutdown.
        if mean(freq_bips_array) < 50:
            change_mod = change_mod + 1
        else:
            change_mod = 0

        print("change_mode: " + str(change_mod))

        stereo_beep(sound_esquerda, sound_direita)
        
        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')

        


try:
    main_program()
finally:
    pipe.stop()

