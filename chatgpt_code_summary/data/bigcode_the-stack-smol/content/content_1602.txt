import sys
import numpy as np 
import cv2

def overlay(img, glasses, pos):
    sx = pos[0]
    ex = pos[0] + glasses.shape[1]
    sy = pos[1]
    ey = pos[1] + glasses.shape[0]


    if sx < 0 or sy < 0 or ex > img.shape[1] or ey > img.shape[0]:
        return

    img1 = img[sy:ey, sx:ex]
    img2 = glasses[:, :, 0:3]
    alpha = 1. - (glasses[:, :, 3] / 255.)

    img1[..., 0] = (img1[..., 0] * alpha + img2[..., 0] * (1. - alpha)).astype(np.uint8)
    img1[..., 1] = (img1[..., 1] * alpha + img2[..., 1] * (1. - alpha)).astype(np.uint8)
    img1[..., 2] = (img1[..., 2] * alpha + img2[..., 2] * (1. - alpha)).astype(np.uint8)


# cam open
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('cam not opened')
    sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30 , (w,h))

# XML file load
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

if face_classifier.empty() or eye_classifier.empty():
    print('xml load error')
    sys.exit()

glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

if glasses is None:
    print('png file load error')
    sys.exit()

ew, eh = glasses.shape[:2]
ex1, ey1 = 240, 300
ex2, ey2 = 660, 300

# Video process
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_classifier.detectMultiScale(frame ,scaleFactor=1.2, minSize=(100,100), maxSize=(400,400))

    for (x, y, w, h) in faces:
        faceROI = frame[y: y+h//2, x: x+w]
        eyes = eye_classifier.detectMultiScale(faceROI)
        
        if len(eyes) != 2:
            continue

        x1 = x + eyes[0][0] + (eyes[0][2] // 2)
        y1 = y + eyes[0][1] + (eyes[0][3] // 2)
        x2 = x + eyes[1][0] + (eyes[1][2] // 2)
        y2 = y + eyes[1][1] + (eyes[1][3] // 2)

        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        fx = (x2 - x1) / (ex2 - ex1)
        glasses2 = cv2.resize(glasses, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)

        pos = (x1 - int(ex1 * fx), y1 - int(ey1 * fx))

        overlay(frame, glasses2, pos)

    out.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()