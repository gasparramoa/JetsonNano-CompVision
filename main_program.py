# -*- coding: utf-8 -*-

#####################################################
##          Door Problem - PointNet Only           ##
#####################################################


import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import os
import time
import torchvision.transforms as transform
import torch

from PIL import Image
from torch.utils import data
# Change here for our database name.
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
from torch.autograd import Variable

from gtts import gTTS
from playsound import playsound
import multiprocessing
from statistics import mean 
import simpleaudio as sa
import wave


def speak(text):
    language = "en"  # pt # language
    file_name = text + ".mp3"  # name of where the file will be saved
    output = gTTS(text=text, lang=language, slow=False)  # configure tts
    output.save(file_name)  # create file
    playsound(file_name)



# Declare RealSense pipeline, encapsulating the actual device and sensors.
pipe = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe.start(config);
align_to = rs.stream.color
align = rs.align(align_to)
threshold = rs.threshold_filter()
threshold.set_option(rs.option.max_distance, 6)


# Stereo Audio:
# get timesteps for each sample, T is note duration in seconds
sample_rate = 44100
T = 0.2272753623188406
t = np.linspace(0, T, T * sample_rate, False)

# Global Variables:
model = ""
classifier = ""
imagemSegmentacao = ""
loader = ""

def inicializar_pointnet():
    
    print("Loading 3D Object Classification Model:")
    start = time.time()
    global classifier
    classifier = PointNetCls(k=3) #IMP, 2 é o número de classes, depois alterar.
    classifier.cuda()
    #classifier.load_state_dict(torch.load('/home/socialab/FCHarDNet/cls_model_40.pth'))
    #classifier.load_state_dict(torch.load('/home/socialab/human_vision/FCHarDNet/cls_model_40.pth'))
    classifier.load_state_dict(torch.load('/home/socialab/JetsonNano-CompVision/model/model_15_0.pth'))    
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

    # Change number of points depending of how the model was trained.
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

    print(pred)

    if str(pred_choice).split("tensor([")[1].split("]")[0] == '0':
        print("\n [CLOSED DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")
        playsound('/home/socialab/JetsonNano-CompVision/sounds/DoorClosed.mp3')
    elif str(pred_choice).split("tensor([")[1].split("]")[0] == '2':
        print("\n [SEMI-OPEN DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")
        playsound('/home/socialab/JetsonNano-CompVision/sounds/DoorSemi.mp3')
    else:
        print("\n [OPEN DOOR] - [" + str(pred_choice).split("tensor([")[1].split("]")[0] + "]\n")
        playsound('/home/socialab/JetsonNano-CompVision/sounds/DoorOpen.mp3')


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
    
    # PRINT MEAN VALUES:
    #print("medias: " + str(meean))

def stereo_beep(left, right):

    if left == 0:
        print("divison 0")

    elif right == 0:
        print("dividion 0")

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



def main_program():


    # Initial sound of the program

    duration = 0.2  # seconds
    freq = 500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 400))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, 600))

    inicializar_pointnet()
    
    modo = 0 # default 0 - street mode

    change_mode = 0

    while True:

        if modo == 0:

            if change_mode == 5:
                change_mode = 0
                modo = 1
                continue

            frames = pipe.wait_for_frames()
            
            threshold_depth = threshold.process(frames.get_depth_frame())
            depth_image2 = np.asanyarray(threshold_depth.get_data())

            start = time.perf_counter()

            processes = []
            i_linha = 0
            f_linha = 60

            manager = multiprocessing.Manager()
            freq_bips = manager.dict()

            for i in range(8):
                p = multiprocessing.Process(name="Processo" + str(i), target=percorre_matriz_imagem2, args=(i, depth_image2, i_linha, f_linha, freq_bips))
                p.start()
                processes.append(p)
                i_linha = i_linha + 60
                f_linha = f_linha + 60

            for process in processes:
                process.join()
                print(process.name)

            to_sound = mean(freq_bips.values())

            freq_bips_array = freq_bips.values()

            esquerda = []
            esquerda.extend((freq_bips_array[0], freq_bips_array[1], freq_bips_array[2], freq_bips_array[3]))
            m_esquerda = mean(esquerda)

            direita = []
            direita.extend((freq_bips_array[4], freq_bips_array[5], freq_bips_array[6], freq_bips_array[7]))
            m_direita = mean(direita)

            # Antes estava, 50, 180, 300 e 500
            # Depois ficou, 250,400, 550 e 700

            if m_direita < 350:
                sound_direita = 100
            elif m_direita < 600:
                sound_direita = 300
            elif m_direita < 900:
                sound_direita = 400
            elif m_direita < 1200:
                sound_direita = 750
            else:
                sound_direita = 0

            if m_esquerda < 350:
                sound_esquerda = 100
            elif m_esquerda < 600:
                sound_esquerda = 300
            elif m_esquerda < 900:
                sound_esquerda = 400
            elif m_esquerda < 1200:
                sound_esquerda = 750
            else:
                sound_esquerda = 0  


            if mean(freq_bips_array) < 50:
                change_mode = change_mode + 1
            else:
                change_mode = 0

            print("change_mode: " + str(change_mode))

            stereo_beep(sound_esquerda, sound_direita)
            
            finish = time.perf_counter()
            print(f'Finished in {round(finish-start, 2)} second(s)')

        elif modo == 1:

            if change_mode == 3:
                change_mode = 0
                modo = 0
                continue

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

