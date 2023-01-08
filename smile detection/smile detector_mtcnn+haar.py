import cv2
import numpy as np
from mtcnn import MTCNN

# Face classifier
face_detector = MTCNN()
# Smile classifier
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Obtain the webcam feed
webcam = cv2.VideoCapture(0)

# Show the current frame (going frame by frame)
while True:
    # Read only the current frame from the webcam
    # Get the tuple containing a Boolean output for whether it's successfully
    # read and then the actual frame
    successful_frame_read, frame = webcam.read()

    # Abort in case there is an error
    if not successful_frame_read:
        break

    # Convert to grayscale (black & white) because the facial recognition ability
    # is increased since RGB has 3 channels while the grayscale has only 1
    # Also, it is a requirement for opencv.
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces & the smiles
    faces = face_detector.detect_faces(frame)

    # Run the face detection within each of those faces
    # We have the x and y coordinates, the width and the height of the rectangle
    for face in faces:
        
        # Draw a rectangle around the face
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        # Slicing the captured webcame frame within the detected face's bounds
        # the_face represents the sub frame - using numpy N-dimensional array slicing
        the_face = frame[y:y+h, x:x+w]

        # Convert the sub frame to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.6, minNeighbors=20)

        # Creating a smile detection loop nested in detecting the face
        #for (x_, y_, w_, h_) in smiles:
            
            # Draw a rectangle around the smile
            #cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 2)

        # If there is at least one smile, display the text "smiling" instead of drawing a rectangle
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
        
    # Show the frame
    cv2.imshow('Smile detector', frame)

    # Keep the window open until a key is pressed
    # Updated every 1ms - close to real time
    cv2.waitKey(1)

# Code executed without any errors
print('Code completed')

# Cleanup
webcam.release()
# cv2.destroyAllWindows()
