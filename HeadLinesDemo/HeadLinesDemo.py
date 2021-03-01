import cv2
import numpy as np
import dlib
from math import hypot
# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
hair = cv2.imread("im.png")

#dimension= hair.shape
#height= hair.shape[0]
#width=hair.shape[1]
#channel=hair.shape[2]
#print("height", height)
#print("w", width)

_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)
# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray_frame, face)

       # for n in range(17, 26):
        #    x = landmarks.part(n).x
         #   y = landmarks.part(n).y
          #  cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        leftSide=(landmarks.part(17).x, landmarks.part(17).y)
        cv2.circle(frame, leftSide, 4, (255, 0, 0), -1)

        rightSide= (landmarks.part(26).x, landmarks.part(26).y)
        cv2.circle(frame, rightSide, 4, (255, 0, 0), -1)

        width= int(hypot(leftSide[0]- rightSide[0], leftSide[1]-rightSide[1]))
        height= int(width * 0.85)

       # cv2.line(frame, leftSide, rightSide, (255, 0, 0))

        cv2.rectangle(frame,leftSide, rightSide, (255, 0, 0))

        hairpng= cv2.resize(hair, (width, height))
         
       
    cv2.imshow("Frame", frame)
#    cv2.imshow("hair", hairpng)
    key = cv2.waitKey(1)
    if key == 27:
        break