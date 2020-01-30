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
# get timesteps for each sample, T is note duration in seconds
sample_rate = 44100
T = 0.2272753623188406
t = np.linspace(0, T, T * sample_rate, False)



def stereo_beep(left, right):

    global t

    # calculate note frequencies
    A_freq = 330

    sample_rate = 22050

    # generate sine wave notes
    A_note = np.sin(A_freq * t * 2 * np.pi)

    # mix audio together
    audio = np.zeros((sample_rate, 2))
    n = len(t)
    offset = 0

    # 1 right side, 0 left side
    audio[0 + offset: n + offset, 0] += right * A_note
    audio[0 + offset: n + offset, 1] += left * A_note

    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))

    # convert to 16-bit data
    audio = audio.astype(np.int16)

    # start playback
    #play_obj = sa.play_buffer(audio, 2, 2, sample_rate)


    # wait for playback to finish before exiting
    #play_obj.wait_done()

    wave_read = wave.open("/home/socialab/JetsonNano-CompVision/beep1_stereo.wav", 'rb')
    audio_data = wave_read.readframes(wave_read.getnframes())

    data = np.fromstring(audio_data, dtype=np.uint16)
    
    data_per_channel = [data[offset::2] for offset in range(2)]

    data[0::2] = 0

    print(len(data))
    for coiso in data:
        print(coiso)

    num_channels = wave_read.getnchannels()
    print(num_channels)
    bytes_per_sample = wave_read.getsampwidth()
    sample_rate = wave_read.getframerate()


    play_obj = sa.play_buffer(data.tostring(), 2, bytes_per_sample, sample_rate)
    play_obj.wait_done()

    '''
    wave_obj = sa.WaveObject.from_wave_file("/home/socialab/JetsonNano-CompVision/beep1.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()
    '''


def speak(text):

    language = "en"  # pt # language
    file_name = text + ".mp3"  # name of where the file will be saved

    output = gTTS(text=text, lang=language, slow=False)  # configure tts
    output.save(file_name)  # create file

    print("oi")

    playsound(file_name)

#os.system("play " + ficheiro)


def percorre_matriz_imagem(matriz):
    
    max_flag = 0
    ele_especial = -1

    flag = 0
    for line in matriz:
        for ele in line:
            if ele == 0:
                continue
            if ele <= 300:
                flag = 1
            if ele <= 180:
                #print(ele)
                flag = 2
            if ele <= 160:
                #print(ele)
                flag = 3
                ele_especial = ele
            
            if flag > max_flag:
                max_flag = flag

    os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 100 * max_flag))

    print(max_flag)
    print(ele_especial)

def bib_sound(number_freq):

    if number_freq > 300:
        print("Não vale a pena tocar som")

    else:
        os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 3000 * (1 / number_freq)))
        time.sleep(0.1)


def beep2000(distance_to_collide):

    print("\ndistance_to_collide")
    print(distance_to_collide)

    if distance_to_collide >= 300:
        print("Não vale a pena tocar som")

    else:
        os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 3000 * (1 / distance_to_collide)))

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
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 400))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, 600))



    #for i in range(12):

    while True:

        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n____________________________________________________________________________")
        print("____________________________________________________________________________")

        program_time = time.time()

        frames = pipe.wait_for_frames()
        
        threshold_depth = threshold.process(frames.get_depth_frame())
        depth_image2 = np.asanyarray(threshold_depth.get_data())

        start = time.perf_counter()


        #percorre_matriz_imagem(depth_image2)

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

        if m_direita < 50:
            sound_direita = 1.0
        elif m_direita < 180:
            sound_direita = 0.75
        elif m_direita < 300:
            sound_direita = 0.5
        elif m_direita < 500:
            sound_direita = 0.25
        else:
            sound_direita = 0

        if m_esquerda < 50:
            sound_esquerda = 1.0
        elif m_esquerda < 180:
            sound_esquerda = 0.75
        elif m_esquerda < 300:
            sound_esquerda = 0.5
        elif m_esquerda < 500:
            sound_esquerda = 0.25
        else:
            sound_esquerda = 0   

        stereo_beep(sound_esquerda, sound_direita)
        
        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')



        image_depth_2 =Image.fromarray(depth_image2)

        image_depth_22 = image_depth_2.rotate(270, expand=True)   


try:
    main_program()
finally:
    pipe.stop()

