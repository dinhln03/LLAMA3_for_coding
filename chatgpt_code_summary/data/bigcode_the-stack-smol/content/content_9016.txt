import cv2
import numpy as np
# import gt_utils


def binarize(img):
    """ Take an RGB image and binarize it.

    :param img: cv2 image
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return bin


def makecoloredlayer(img, mask, color=(0, 255, 0)):
    """ Create an image based on provided mask, dimensions and color (default is red)

    :param mask: binary mask defining the zone needing color, the rest will be black
    :param img: numpy.array used for dimensions
    :param color: 3 dimension tuple (B,G,R)
    :return:
    """
    coloredlayer = np.zeros(img.shape, dtype="uint8")
    # cv2.rectangle(img, (x,y), (x,y), color, -1)
    cv2.rectangle(coloredlayer, (0, 0), (img.shape[1], img.shape[0]), color, -1)
    maskedlayer = cv2.bitwise_or(img, coloredlayer, mask=mask)
    return maskedlayer


def makemarkedmask(maskA, maskB):
    """ create a new mask based on existing image and coming mask

    :param maskA: binary image
    :param maskB: binary image
    :return: binary image
    """
    inter = cv2.bitwise_xor(maskA, maskB)
    inter = cv2.bitwise_and(inter, maskB)
    inter = cv2.bitwise_xor(inter, maskB)
    markedmask = cv2.bitwise_not(inter)
    return markedmask


def applymark(img, mask):
    """ Apply a mask to an image to keep only active cells.

    :return: image
    """
    img = cv2.bitwise_and(img, img, mask=mask)
    return img


def makeannotatedimage(masks, colors):
    """ Combine layers and create an image combining several annotations

    :param masks: list of images used as masks
    :param colors: list of colors
    :return:
    """
    if len(masks) != len(colors):
        print("Error: annotation and colors do not match.")
        return False
    else:
        if len(masks) == 0:
            print("Error: no mask to combine.")
            return False
        else:
            # combo is canvas (y,x,3)
            combo = np.zeros(masks[0].shape, dtype='uint8')
            i = 0
            # binarize images to make masks
            bins = []
            for mask in masks:
                bin = binarize(mask)
                bins.append(bin)

            # isolate first mask/layer from the rest
            firstmask = bins[:1]
            combo = makecoloredlayer(combo, firstmask[0], colors[i])

            if len(bins) > 1:
                other_masks = bins[1:]
                # adding other layers
                for mask in other_masks:
                    i += 1
                    newmask = binarize(combo)
                    markedout = makemarkedmask(newmask, mask)
                    combo = applymark(combo, markedout)
                    newlayer = makecoloredlayer(combo, mask, colors[i])
                    combo = cv2.bitwise_or(combo, newlayer)

            return combo
