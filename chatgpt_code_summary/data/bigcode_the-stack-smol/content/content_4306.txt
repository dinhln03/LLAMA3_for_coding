# Thai Thien
# 1351040

import pytest
import cv2
import sys
import sys, os
import numpy as np
import upload

# make sure it can find matcher.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import util
from matcher import Matcher

# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from detector import Detector


prefix = './image/meowdata/'
chuot1_path =  prefix + 'chuot1.jpg'
chuot2_path =  prefix +'chuot1.jpg'
chuot3_path =  prefix +'chuot1.jpg'

dau1_path =  prefix +'dau1.jpg'
dau2_path =  prefix +'dau2.jpg'
dau3_path =  prefix +'dau3.jpg'
dau4_path =  prefix +'dau4.jpg'

keyboard1_path = prefix +'keyboard1.jpg'
keyboard2_path = prefix +'keyboard2.jpg'
keyboard3_path = prefix +'keyboard3.jpg'
keyboard4_path = prefix +'keyboard4.jpg'

chuot1 = cv2.imread(chuot1_path)
chuot2 = cv2.imread(chuot2_path)
chuot3 = cv2.imread(chuot3_path)

dau1 = cv2.imread(dau1_path)
dau2 = cv2.imread(dau2_path)

keyboard1 = cv2.imread(keyboard1_path)
keyboard2 = cv2.imread(keyboard2_path)

isUpload = False
class TestMatcher():

    def test_matches_dog_sift(self):
        _matcher = Matcher()
        _name = 'chuot1_2_dog_sift'
        _file  = './output/'+_name+'.png'
        matches, result = _matcher.dog_match(chuot1, chuot2, 20)
        cv2.imwrite(_file, result)
        if (isUpload):
            upload.imgur(_file,_name)

        _name = 'keyboard1_2_dog_sift'
        _file  = './output/'+_name+'.png'
        matches, result = _matcher.dog_match(chuot1, chuot2, 20)
        cv2.imwrite(_file, result)
        if (isUpload):
            upload.imgur(_file, _name)

