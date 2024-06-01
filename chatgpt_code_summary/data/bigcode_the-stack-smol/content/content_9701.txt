import cv2


def mean(image):
    return cv2.medianBlur(image, 5)


def gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
