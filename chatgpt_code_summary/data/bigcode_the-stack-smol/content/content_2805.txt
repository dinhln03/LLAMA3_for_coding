#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import tensorflow as tf


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers


# ### Load the Training Data

# In[4]:

# curwd = str(os.getcwd())
# targetwd = '\\data\\train'
# path_train = curwd + targetwd
path_train = '...s\\Documents\\whale_identification\\whale_identification\\data\\train\\'
train = [os.path.join(path_train,f) for f in os.listdir(path_train) if f.endswith('.jpg')]



# In[6]:


train_labels = pd.read_csv("df_train.csv")


# In[7]:


train_labels.head()


# In[8]:


unique_whales = train_labels['Id'].unique()
len(unique_whales)


# ### Train-Validation Split

# In[9]:


def train_valid_split(df):

    # find unique categories of whales in our dataframe
    unique_whales = train_labels['Id'].unique()

    # map the images to categories
    mapping = {}
    for whale in unique_whales:
        lst_of_images = list(train_labels[train_labels['Id'] == whale]['Image'].values)
        mapping[whale] = lst_of_images

    # perform manual train/validation split to ensure balanced data in both sets (i.e. all categories are represented)
    train_revised = []
    valid_revised = []

    for v in mapping.values():
        cut = int(0.2*len(v)) # sample & 80-20 split
        cut2 = int(0.25*len(v))
        tr = v[:cut]
        val = v[cut:cut2]
        train_revised.append(tr)
        valid_revised.append(val)

    return train_revised, valid_revised

def train_valid_dict_generator(train_list, valid_list, df):
    
    # create a dictionary mapping new training set to correct labels
    train_df = {}

    for i in train_list:
        for j in i:
            lbl = df[df['Image'] == j]['Id'].values[0]
            train_df[j] = lbl
    
    # create a dictionary mapping new validation set to correct labels
    valid_df = {}

    for i in valid_list:
        for j in i:
            lbl = df[df['Image'] == j]['Id'].values[0]
            valid_df[j] = lbl

    return train_df, valid_df