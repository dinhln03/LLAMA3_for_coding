import cv2
import os

image = cv2.imread("/content/drive/My Drive/DIC_personal/data/face.jpg")
cascade = cv2.CascadeClassifier("/content/drive/My Drive/DIC_personal/haarcascades/haarcascade_upperbody.xml")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_list = cascade.detectMultiScale(image)
#face_list = cascade.detectMultiScale(image,scaleFactor=1.2, minNeighbors=2, minSize=(1,1))

color = (0, 0, 255)

if len(face_list)>0:
    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(image,(x,y),(x+w, y+h), color, thickness=2)
else:
    print("No human")
    
cv2.imshow('Frame',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

https://qiita.com/PonDad/items/6f9e6d9397951cadc6be