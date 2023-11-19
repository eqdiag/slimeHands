import cv2 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# STEP 1: Import the necessary modules.

# STEP 2: Create an HandLandmarker object.
#base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#options = vision.HandLandmarkerOptions(base_options=base_options,
                                       #num_hands=2)
#detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.z
#image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect hand landmarks from the input image.
#detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
#annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv.mshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

DESIRED_SHAPE = (100,100)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()