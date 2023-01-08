import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

    # Detect the faces
    faces = face_detector.detectMultiScale(frame_grayscale)
    
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
