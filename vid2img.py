#!/usr/bin/python3

import cv2 as cv
import sys


args = sys.argv

USAGE = f'{args[0]} <video file relative to ./videos> <frame directory relative to ./frames>' 

if(len(args) != 3):
    print(USAGE)
    quit()


VIDEO_NAME = './videos/' + args[1]
FRAME_DIR = './frames/' + args[2]



#Videos to frames
capture = cv.VideoCapture(VIDEO_NAME)
frameNr = 0

while (True):
 
    success, frame = capture.read()
 
    if success:
        cv.imwrite(f'{FRAME_DIR}/frame_{frameNr}.jpg', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()