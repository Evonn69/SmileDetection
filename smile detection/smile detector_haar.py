import cv2
from PIL import Image

image1 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image1.jpg")
image2 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image2.jpg")
image3 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image3.jpg")
image4 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image4.jpg")

image1 = cv2.resize(image1, None, fx=0.50, fy=0.50)
image2 = cv2.resize(image2, None, fx=0.50, fy=0.50)
image3 = cv2.resize(image3, None, fx=0.50, fy=0.50)
image4 = cv2.resize(image4, None, fx=0.50, fy=0.50)

image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
image4_gray = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

# The Haarcascade algorithm
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(image2_gray, scaleFactor=1.2, minNeighbors=5)
image = image2.copy()

# Draw rectangles on each face
for (x, y, w, h) in faces:
    image = cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)

# Convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image
Image.fromarray(image).save('image22.jpg') 
