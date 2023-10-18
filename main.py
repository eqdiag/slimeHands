import cv2 as cv
import numpy as np
import os


#View frames in grayscale, convert them to vecs
for file in os.listdir("./frames"):
    path = "./frames/" + file
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    x = np.array(img)
    x = x.flatten()
    x = np.reshape(x,(x.shape[0],1))
    #print(train.shape)
    cv.imshow('graycsale image',img)
    cv.waitKey(0)


#Videos to frames
''' 
capture = cv.VideoCapture('./videos/1.mp4')
frameNr = 0

while (True):
 
    success, frame = capture.read()
 
    if success:
        cv.imwrite(f'./frames/frame_{frameNr}.jpg', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()
'''




