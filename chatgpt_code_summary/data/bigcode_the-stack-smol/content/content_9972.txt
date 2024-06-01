import cv2
import os
import numpy as np
from image_processor import process_image
from processor_properties import ProcessorProperties
import time


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)


    def snapshot(self):
        ret, frame = self.cap.read()
        return frame


if __name__ == '__main__':
    camera = Camera()
    while True:
        frame = camera.snapshot()
        props = ProcessorProperties()
        # props.brightness_factor.update(1.5)
        # props.contrast_factor.update(1.5)
        # props.scaling_factor.update(3.0)
        frame = process_image(frame, props)
        cv2.imshow('image', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            image_path = os.path.join("testimgs", "%s.jpg" % timestr)
            cv2.imwrite(image_path, frame)
            print "save %s" % image_path