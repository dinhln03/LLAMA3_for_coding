#-*-coding:utf-8-*-

import numpy as np
import cv2
import gc
from tqdm import tqdm

def watershed(opencv_image):
    top_n_label = 2

    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    print('convert gray end')

    gray[gray == 0] = 255

    _, cvt_img = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    del(gray)
    print('threshold end')


    ret, markers = cv2.connectedComponents(cvt_img)
    print('connectedComponents end')

    label_dict = dict()
    for i in tqdm(range(ret)):
        if i == 0:
            continue
        label_dict[i] = len(markers[markers == i])
    sort_label_list = sorted(label_dict.items(), key=lambda item: item[1], reverse=True)
    print('label end')

    result = np.zeros(markers.shape)
    for ins in tqdm(sort_label_list[:top_n_label]):
        result[markers == ins[0]] = 255


    print(result.shape)

    print('top n label end')
    del(ret)
    del(markers)
    del(sort_label_list)
    del(label_dict)
    del(cvt_img)


    return result