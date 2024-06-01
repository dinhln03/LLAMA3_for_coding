#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from PyQt5.QtGui import QImage, qRgb, QPixmap
import numpy as np
import numpy as np

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def toQImage(data, copy=True):
    if data is None:
        return QImage()

    data = data.copy()

    data[data>255] = 255
    data[data<0] = 0
    data = data.astype(np.uint8)

    if data.dtype == np.uint8:
        if len(data.shape) == 2:
            qim = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(data.shape) == 3:
            if data.shape[2] == 1:
                qim = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Grayscale8)
                return qim.copy() if copy else qim
            if data.shape[2] == 3:
                qim = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_RGB888)
                return qim.copy() if copy else qim
            elif data.shape[2] == 4:
                qim = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim
            else:
                raise Exception("Conversion of %d channel array to QImage not implemented" % data.shape[2])

    raise Exception("Conversion of %d dimension array to QImage not implemented" % len(data.shape))

def toQPixmap(data):
    if data is None: return QPixmap()
    elif isinstance(data, QPixmap):  return data
    elif isinstance(data, QImage):  QPixmap.fromImage(data)
    elif hasattr(data, 'pixmap'): return data.pixmap()
    else: return QPixmap.fromImage(toQImage(data))

def qPixmapToNumpy(pixmap):
    image = pixmap.toImage()
    image = image.convertToFormat(QImage.Format.Format_RGB32)

    width = image.width()
    height = image.height()

    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr[:, :, 0:3].copy()
