#!/usr/bin/env python3

from helper import lemniscate, saveImage, getImageName
from math import pi
from random import uniform

IMAGE_WIDTH_IN = 5
IMAGE_HEIGHT_IN = 2
DPI = 400

def makeImage(height, width, imageName):
    k = 1.9
    numSamples = 100000000
 
    imageWidth = width
    imageHeight = height

    data = [0] * imageHeight * imageWidth

    cx = imageWidth/2
    cy = imageHeight/2
    radius  = imageWidth/2 * .95

    for i in range(numSamples):
        # need to increase range for complete curve
        theta = uniform(0, 10 * pi)
        
        pA_x, pA_y = lemniscate(theta, radius, cx, cy)
        pB_x, pB_y = lemniscate(k * theta, radius, cx, cy)
        
        # pick a random point on the line segment [pA, pB]
        r = uniform(0, 1)
        pC_x = (1 - r) * pA_x + r * pB_x
        pC_y = (1 - r) * pA_y + r * pB_y
         
        i = int(pC_x + .5)
        j = int(pC_y + .5)
        
        data[j * imageWidth + i] += 1
        
    saveImage(data, imageName, imageWidth, imageHeight,
              bg=[255, 255, 255], fg=[221, 100, 0], alphaMultiplier=18)
    

if __name__ == "__main__":
    imageName = getImageName(__file__)
    
    width = IMAGE_WIDTH_IN * DPI
    height = IMAGE_HEIGHT_IN * DPI
    makeImage(height, width, imageName)
