import cv2 
import mediapipe as mp
import numpy as np
from os import listdir
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands






def get_imgs_from_dir(path):
  imgs = []
  for file in listdir(path):
      imgs.append(path + "/" + file)
  return imgs

def landmarks_to_big_vec(landmarks):
  col = []
  for mark in mp_hands.HandLandmark:
    col.append(landmarks.landmark[mark].x)
    col.append(landmarks.landmark[mark].y)
    col.append(landmarks.landmark[mark].z)
  return np.asarray(col)

def get_closest_idx(test_hand_entry,test_hand_data_set):
  num_hands = test_hand_entry[0]
  #For now, if no hands detected, grab first image without hands
  if(num_hands == 0):
    idx = 0
    for entry in test_hand_data_set:
      if(entry[0] == 0):
        return idx 
      idx+=1
  if(num_hands == 1):
    hand_type = test_hand_entry[1][0]
    minloss = 100000000000
    minidx = 0
    idx = 0
    for entry in test_hand_data_set:
      loss = 0
      if(entry[0] == 1):
        if(entry[1][0] == hand_type):
          v0 = test_hand_entry[1][1]
          v1 = entry[1][1]
          dv = v1 - v0
          loss += np.vdot(dv,dv)
          if(loss < minloss):
            minloss = loss
            minidx = idx
      idx+=1
    return minidx
  #todo: handle two hand case
  


imgs = get_imgs_from_dir("./frames/vid1")
print(len(imgs))


#tuple [num hands,(hand type,data),(hand type, data),...,image data)

HAND_STUFF = []

with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as model:
  for idx, file in enumerate(imgs):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    hand_meta_data = results.multi_handedness
    hand_landmarks = results.multi_hand_landmarks
    hand_info = []

    if(hand_meta_data == None):
      hand_info.append(0)
      hand_info.append(image)
    elif(len(hand_meta_data) == 1):
      hand_info.append(1)
      hand_type = hand_meta_data[0].classification[0].label
      hand_vec = landmarks_to_big_vec(hand_landmarks[0])
      hand_info.append((hand_type,hand_vec))
      hand_info.append(image)
    else: #len(hand_meta_data) == 2 case
      hand_info.append(2)
      hand_type0 = hand_meta_data[0].classification[0].label
      hand_type1 = hand_meta_data[1].classification[0].label
      hand_vec0 = landmarks_to_big_vec(hand_landmarks[0])
      hand_vec1 = landmarks_to_big_vec(hand_landmarks[1])
      hand_info.append((hand_type0,hand_vec0))
      hand_info.append((hand_type1,hand_vec1))
      hand_info.append(image)
   
    HAND_STUFF.append(hand_info)



# For webcam input:

prev_idx = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    #needs img in rgb for model to work
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #run through hand modle
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    #back to bgr for cv for display properly
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    hand_meta_data = results.multi_handedness
    hand_landmarks = results.multi_hand_landmarks

    hand_info = []

    if(hand_meta_data == None):
      hand_info.append(0)
      hand_info.append(image)
    elif(len(hand_meta_data) == 1):
      hand_info.append(1)
      hand_type = hand_meta_data[0].classification[0].label
      hand_vec = landmarks_to_big_vec(hand_landmarks[0])
      hand_info.append((hand_type,hand_vec))
      hand_info.append(image)
    else: #len(hand_meta_data) == 2 case
      hand_info.append(2)
      hand_type0 = hand_meta_data[0].classification[0].label
      hand_type1 = hand_meta_data[1].classification[0].label
      hand_vec0 = landmarks_to_big_vec(hand_landmarks[0])
      hand_vec1 = landmarks_to_big_vec(hand_landmarks[1])
      hand_info.append((hand_type0,hand_vec0))
      hand_info.append((hand_type1,hand_vec1))
      hand_info.append(image)

    idx = get_closest_idx(hand_info,HAND_STUFF)
    if(idx == None):
      idx = prev_idx
    else:
      prev_idx = idx

    guess = HAND_STUFF[idx]
    guess_image = guess[guess[0] + 1]

    #print(guess_image.shape)

    down_img = cv2.resize(image,(guess_image.shape[1],guess_image.shape[0]),interpolation = cv2.INTER_LINEAR)
    #print(down_img.shape)

    #Put skeleton hands on top of img
    if results.multi_hand_landmarks:
      for some_hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(down_img,some_hand,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())

    side_by_side_img = np.concatenate((guess_image,down_img),axis = 1)

    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('hand tracker', cv2.flip(side_by_side_img, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()