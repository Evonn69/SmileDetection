# Import the libraries
import dlib
import cv2

# Obtaining the image by opencv function and resizing it for the sake of the algorithm then turning it from RGB into grayscale
image1 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image4.jpg")
image1 = cv2.resize(image1, None, fx=0.50, fy=0.50)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# The face classifier
face_detector = dlib.get_frontal_face_detector()

# Detect faces
detections = face_detector(image1, 1)
image = image1.copy()

# Draw rectangles on each face
for (i, rect) in enumerate(detections):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image
cv2.imshow("Image", image)
