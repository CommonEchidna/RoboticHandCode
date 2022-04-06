import cv2
from matplotlib.pyplot import pink
import mediapipe as mp
import numpy as np
import serial

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
def getclenched(idx,hand_landmarks):
    base = np.array([hand_landmarks.landmark[1+idx].x,hand_landmarks.landmark[idx+1].y,hand_landmarks.landmark[1+idx].z])-np.array([hand_landmarks.landmark[2+idx].x,hand_landmarks.landmark[2+idx].y,hand_landmarks.landmark[2+idx].z])
    end = np.array([hand_landmarks.landmark[3+idx].x,hand_landmarks.landmark[3+idx].y,hand_landmarks.landmark[3+idx].z])-np.array([hand_landmarks.landmark[4+idx].x,hand_landmarks.landmark[4+idx].y,hand_landmarks.landmark[4+idx].z])

    thumbclenched = np.dot(base,end)/(np.linalg.norm(base)*np.linalg.norm(end))
    return thumbclenched
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("No frame detected")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    coordlist = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        for ids, landmrk in enumerate(hand_landmarks.landmark):
          if ids==0:
            cx, cy,cz = landmrk.x, landmrk.y,landmrk.z
            cx, cy, cz = round(50*(cx+1)/2),round(50*(cy+1)/2),round(50*(cz+1)/2)
            indexclenched = round((getclenched(4,hand_landmarks)+1)/2)
            middleclenched = round((getclenched(8,hand_landmarks)+1)/2)
            ringclenched = round((getclenched(12,hand_landmarks)+1)/2)
            pinkyclenched = round((getclenched(16,hand_landmarks)+1)/2)
            arr = [indexclenched,middleclenched,ringclenched,pinkyclenched]
            avg = round(sum(arr)/4)
            coordlist.append([avg, cx, cy,cz])
            ser = serial.Serial(port="COM4",baudrate = 115200,write_timeout = 0)
            ser.write(coordlist)
            
            


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()