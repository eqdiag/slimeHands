#!/usr/bin/python3

import cv2 as cv
import numpy as np
import os

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.tasks.python import vision



MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

#Step 2
# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
mp_drawing = mp.solutions.drawing_utils


mp.Image()
image = mp.Image.create_from_file("./frames/vid1/frame_12.jpg")
print("hey")
print(type(image))
detection_result = detector.detect(image)

#print(detection_result)

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

print(type(annotated_image))
print(annotated_image.shape)

cv.imshow("test",cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

FRAME_DIR = "./frames/vid1"

NUM_NEIGHBORS = 10

DESIRED_SHAPE = (100,100)
DISPLAY_SHAPE = (200,200)
DATA_LEN = DESIRED_SHAPE[0]*DESIRED_SHAPE[1]
TRAINING_FRAC = 0.7


data = []
rgb_images = []

print("Loading in images from `frames` dir...")

#Read all images in "frames" dir as grayscale into rows of matrix of size (num images x DATA_LEN)
for file in os.listdir(FRAME_DIR):
    path = FRAME_DIR +'/' +file
    
    rgb_img = cv.imread(path,cv.COLOR_BGR2RGB)
    rgb_images.append(rgb_img)
    downsampled_img = cv.resize(rgb_img,DESIRED_SHAPE,interpolation = cv.INTER_LINEAR)
    grey_img = cv.cvtColor(downsampled_img,cv.COLOR_RGB2GRAY)
    x = np.array(grey_img).flatten().reshape((1,DATA_LEN))
    #cv.imshow('im2',np.reshape(x,DESIRED_SHAPE))
    #cv.waitKey(0)

    data.append(x)



print("Formatting data to be squeezed into kNN model...")

#Prepare data for kNN model
NUM_IMAGES = len(data)
TRAINING_SIZE = int(TRAINING_FRAC*NUM_IMAGES)
VALIDATE_SIZE = NUM_IMAGES - TRAINING_SIZE

#Kinda abusive use of kNN, using index as category
responses = np.arange(TRAINING_SIZE).reshape((TRAINING_SIZE,1)).astype(np.float32)

X = np.zeros((NUM_IMAGES,DATA_LEN)).astype(np.float32)
for i in range(NUM_IMAGES):
    X[i] = data[i]



print("Winding up the greasy cogs of the kNN distance machine...")


#Set up kNN model
knn = cv.ml.KNearest_create()
knn.train(X[0:TRAINING_SIZE,:], cv.ml.ROW_SAMPLE,responses)

print("Computing closest slimes...")

ret, results, neighbours ,dist = knn.findNearest(X[TRAINING_SIZE:,:],NUM_NEIGHBORS)

print(neighbours.shape)

#Controls number of random images to test out at the end
NUM_RANDOM = 0

print(f"Now let's show {NUM_RANDOM} random images and their closest images from the dataset...fingers crossed...")


for i in range(NUM_RANDOM):
    #Pick a random image in validation set and see what the closest image is
    rand_validation_idx = np.random.randint(0,VALIDATE_SIZE) 
    rand_img = cv.resize(rgb_images[rand_validation_idx + TRAINING_SIZE],DISPLAY_SHAPE,interpolation = cv.INTER_LINEAR)


    #0th index is closest
    closest_img_index = int(neighbours[rand_validation_idx][0])
    closest_img = cv.resize(rgb_images[closest_img_index],DISPLAY_SHAPE,interpolation = cv.INTER_LINEAR)

    side_by_side_img = np.concatenate((rand_img,closest_img),axis = 1)

    cv.imshow(f'LEFT =  Random (IDX = {rand_validation_idx}) | RIGHT = Closest (IDX = {closest_img_index})',side_by_side_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


#Live capture (using iriun)
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    key = cv.waitKey(1)
    downsampled_img = cv.resize(frame,DESIRED_SHAPE,interpolation = cv.INTER_LINEAR)
    grey_img = cv.cvtColor(downsampled_img,cv.COLOR_RGB2GRAY)
    x = grey_img.flatten().reshape((1,DATA_LEN)).astype(np.float32)
    ret, results, neighbours ,dist = knn.findNearest(x,NUM_NEIGHBORS)
    idx = int(neighbours[0][0])

    closest_img = cv.resize(rgb_images[idx],DISPLAY_SHAPE,interpolation = cv.INTER_LINEAR)

    downsampled_capture = cv.resize(frame,DISPLAY_SHAPE,interpolation = cv.INTER_LINEAR)
    side_by_side_img = np.concatenate((downsampled_capture,closest_img),axis = 1)


    cv.imshow('frame',side_by_side_img)
    if key == 113:
        break

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




