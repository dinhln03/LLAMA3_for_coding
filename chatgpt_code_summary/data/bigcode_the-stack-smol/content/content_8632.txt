'''
This module has all relevant functions to make predictions using a previously
trained model.
'''
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

model = load_model('../models/app_ban_ora_selftrained')

def decode_prediction(prediction):
    '''
    Decodes predictions and returns a result string.
    '''
    if np.where(prediction == np.amax(prediction))[1] == 2:
        prob_orange = round(prediction[0][2] * 100, 2)
        label = f"I am {prob_orange} % sure this is an orange \N{tangerine}!"
    if np.where(prediction == np.amax(prediction))[1] == 1:
        prob_banana = round(prediction[0][1] * 100, 2)
        label = f"I am {prob_banana} % sure this is a banana \N{banana}!"
    if np.where(prediction == np.amax(prediction))[1] == 0:
        prob_apple = round(prediction[0][0] * 100, 2)
        label = f"I am {prob_apple} % sure this is an apple \N{red apple}!"
    return label 

def predict(frame):
    '''
    Takes a frame as input, makes a prediction, decoodes it
    and returns a result string.
    '''
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    label = decode_prediction(prediction)
    return label
