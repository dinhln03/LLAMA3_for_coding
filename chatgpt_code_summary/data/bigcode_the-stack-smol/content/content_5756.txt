import cv2
import os
import numpy as np
import faceReacognition as fr

test_img = cv2.imread('b.jpg')

faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected ",faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w, y+h),(0,0,255),thickness=1)

resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow('faces',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
