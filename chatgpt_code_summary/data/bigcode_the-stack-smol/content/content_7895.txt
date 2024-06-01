#!/usr/bin/env python

import os
import sys

basepath = os.path.dirname(os.path.abspath(__file__))
basepath = os.path.abspath(os.path.join(basepath, os.pardir))

wavpath = os.path.join(basepath, "spectro", "tchaikovsky.wav")

sys.path.append(basepath)

from utils import PhysicalKeyboard
from spectro import Spectro

kbd = PhysicalKeyboard()
spectro = Spectro(kbd)

spectro.play(wavpath)
