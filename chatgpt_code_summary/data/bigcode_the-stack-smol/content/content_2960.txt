import numpy as np 
import cv2

# To capture webcam live stream, simply change the following line to: cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./assets/video.mp4')

while (True):
    # Capture frame by frame
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv (hue sat value) for the color red
    lower_color = np.array([150, 150, 50])
    upper_color = np.array([180, 255, 150])

    # mask will be anything between range *lower_color to upper_color (Red)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    res = cv2.bitwise_and(frame, frame, mask = mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (200, 255, 0), 4)

    if len(contours) > 0:      
        cv2.putText(mask, 'Relavante Object Detected', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()