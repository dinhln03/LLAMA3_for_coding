import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#直方图反向投影

def bitwise_and():
    small = cv.imread("C:/1/image/small.jpg")
    big = cv.imread("C:/1/image/big.jpg")
    small_hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    big_hsv = cv.cvtColor(big, cv.COLOR_BGR2HSV)

    """
    h,s,v = cv.split(small_hsv)
    print(h)
    print(s)
    print(v)
    """

    lower_hsv = np.array([1, 120, 240])
    upper_hsv = np.array([4, 160, 255])
    mask = cv.inRange(big_hsv, lower_hsv, upper_hsv)
    dest = cv.bitwise_and(big_hsv, big_hsv, mask=mask)
    cv.imshow('mask', dest)
    cv.imshow('video', big)


def back_projection_demo():
    sample = cv.imread("C:/1/image/small.jpg")
    target = cv.imread("C:/1/image/big.jpg")
    roi_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)

    #show images
    cv.imshow("sample",sample)
    cv.imshow("target",target)

    roiHist = cv.calcHist([roi_hsv],[0,1],None,[32,32],[0,180,0,256])            #求出样本直方图
    cv.normalize(roiHist,roiHist,0,256,cv.NORM_MINMAX)                           #直方图归一化
    dest = cv.calcBackProject([target_hsv],[0,1],roiHist,[0,180,0,256],1)        #直方图反向投影
    cv.imshow("back_projection_demo", dest)


def hist2d_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image],[0,1],None,[32,32],[0,180,0,256])
    # cv.imshow("hist2d_demo",hist)
    plt.imshow(hist,interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()


src = cv.imread("C:/1/1.jpg")
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# cv.imshow("input_image",src)

bitwise_and()

cv.waitKey(0)
cv.destroyAllWindows()
