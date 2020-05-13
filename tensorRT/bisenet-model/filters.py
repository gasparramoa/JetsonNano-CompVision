import cv2 as cv
import numpy as np
img = cv.imread('/home/socialab/human_vision/TorchSeg/model/bisenet/cityscapes.bisenet.R18.speed/image.png/0000719.png',cv.IMREAD_GRAYSCALE)
kernel = np.ones((10,10),np.uint8)
import time


inicio = time.time()


mask = cv.inRange(img, 199, 201)

closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Só para ter feedback
cv.imwrite("mask.png",closing)

contours,hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

if len(contours) != 0:
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    height, width = img.shape

    print(x, y, w, h)
    print(width)

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


end = time.time()
print("Tempo: " + str(end-inicio))

img = cv.imread('original.png')

cv.imwrite("filtered_image.png",img[y_save: yf_save, x_save: xf_save])