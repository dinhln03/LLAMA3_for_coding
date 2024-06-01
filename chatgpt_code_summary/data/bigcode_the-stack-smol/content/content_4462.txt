import os
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

file_index = 0

if not os.path.exists("calibration_data/calibration_frames/"):
    os.makedirs("calibration_data/calibration_frames/")

while True:
    ret, frame = cap.read()

    # display the resulting frame
    cv2.imshow("frame", frame)
    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        filename = "calibration_data/calibration_frames/frame_" + str(file_index) + str(time.time()) + ".png"
        cv2.imwrite(filename, frame)
        print(f"saved frame: {filename}")
        file_index += 1

cv2.destroyAllWindows()
cap.release()
