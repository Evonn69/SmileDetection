import cv2
import numpy as np
import dlib

face_detector = dlib.get_frontal_face_detector()
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame, 1)

    for (i, rect) in enumerate(faces):
        x = rect.left()
        y = rect.top()
        w = rect.right()-x
        h = rect.bottom()-y
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        the_face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.6, minNeighbors=20)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
        
    cv2.imshow('Smile detector', frame)
    cv2.waitKey(1)

webcam.release()
