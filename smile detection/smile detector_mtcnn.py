# Import the libraries
import cv2
from mtcnn import MTCNN

# Obtaining the image by opencv function and resizing it for the sake of the algorithm then turning it from RGB into grayscale
image1 = cv2.imread("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/smile detection/group photos/image4.jpg")
image1 = cv2.resize(image1, None, fx=0.48, fy=0.48)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# The face classifier
face_detector = MTCNN()

# Detect faces
detections = face_detector.detect_faces(image1)
image = image1.copy()

# Draw rectangles on each face
for face in detections:
    x, y, w, h = face['box']
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the image
cv2.imshow("Image", image)

# Cleanup
cv2.waitKey(0)
cv2.destroyAllWindows()
