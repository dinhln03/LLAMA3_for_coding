from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(".."))
from termux2d import Canvas, Palette, line, animate, COLOR_RED, COLOR_WHITE
import math


def __main__():
    i = 0
    height = 40

    while True:
        frame = []

        frame.extend([(coords[0],coords[1],COLOR_WHITE) for coords in
                      line(0,
                           height,
                           180,
                           math.sin(math.radians(i)) * height + height)])

        frame.extend([(x/2, height + math.sin(math.radians(x+i)) * height, COLOR_WHITE)
                      for x in range(0, 360, 2)])

        yield frame

        i += 2



if __name__ == '__main__':
    animate(Canvas(), Palette(), __main__, 1./60)
