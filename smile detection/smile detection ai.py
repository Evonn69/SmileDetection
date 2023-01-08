#Importing necessary libraries
import cv2
import tensorflow as tf
import numpy as np

#Loading the pre-trained model
model = tf.keras.models.load_model('smile_detector_model.h5')

#Setting up the webcam
webcam = cv2.VideoCapture(0)

#Running the loop to continuously detect smiles
while True:
    #Reading the frame from the webcam
    frame = webcam.read()[1]
    
    #Resizing the frame and converting it to a grayscale image
    frame = cv2.resize(frame, (48,48))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #Normalizing the frame
    img_array = np.array(gray).reshape(1, 48, 48, 1)
    img_array = img_array/255.0
    
    #Predicting if the person is smiling or not
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    if result == 0:
        prediction_text = 'No Smile'
    else:
        prediction_text = 'Smile'
        
    #Showing the frame with the prediction text
    cv2.putText(frame, prediction_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.imshow('Smile Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releasing the webcam
webcam.release()
cv2.destroyAllWindows()
