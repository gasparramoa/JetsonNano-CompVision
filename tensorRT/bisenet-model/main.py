import engine as eng
import inference as inf
import tensorrt as trt
import numpy as np
from PIL import Image
import cv2
import os
import time

#https://we.tl/t-X1lk9z4lEg

input_file_path = "/test-images/0000207.png"
serialized_plan_fp32 =  "bisenet.trt"
HEIGHT = 640
WIDTH = 480

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


start = time.time()
print("\nLoading Engine (TensorRT Model)")

im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225])

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
end = time.time()
print("Loading Time: " + str(end-start))

def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def color_map_Ramoa(output):

    out_col = np.zeros(shape=(80, 60), dtype=(np.uint8, 3))

    for x in range(60):
        for y in range(80):

            if (np.argmax(output[y, x] ) == 0.0):
                out_col[y,x] = (0, 200, 200)
            else:
                out_col[y, x] = (200,0,0)
    return out_col 

flag = 100





image = np.asarray(Image.open(input_file_path))

im = np.array(image, dtype=np.float32, order='C')

im = normalize(im, im_mean, im_std)

im = im.transpose((2, 0, 1))



#im = sub_mean_chw(im)
#print(im[0])
#print(im.shape)
end = time.time()
print("\nImage Process Time: " + str(end-start))

start_big = time.time()

while flag > 0:

    flag = flag - 1

    start = time.time()
    pred = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
    pred = pred.permute(1, 2, 0)
    pred = pred.cpu().numpy()
    end = time.time()
    print("\nInference Time: " + str(end-start))

end = time.time()   
print("\nTotal time: " + str(end-start_big))

time_per_inference = (end-start_big) / 100

print("\nTime per inference = " + str(time_per_inference))

start = time.time()
pred = cv2.resize(pred, (480, 640), interpolation=cv2.INTER_LINEAR)
pred = pred.argmax(2)
end = time.time()
print("\nResize Time: " + str(end-start))



#fn = 'imagemFINAL.png'
#pred[pred < 2] = 200
#cv2.imwrite(os.path.join(fn), pred)

