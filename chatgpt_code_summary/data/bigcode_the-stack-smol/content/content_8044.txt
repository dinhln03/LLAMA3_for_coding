#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:14 2018

@author: magalidrumare
@ copyright  https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Use of a pre-trained convnet : VGG16

# An effective approach to deep learning on small image dataset is to leverage a pre-trained network
# A saved network trained on a large dataset for classification task. 
# -> ImageNet (1.4 million labeled images and 1000 different classes),VGG, ResNet, INception, Xception, etc...

# Part 1-Take the convolutional base of a previous trained network and running the data throught it 
# Part 2- Train a new classifier on top of the output 

# Why not reuse the classifier on the top? 
# ->The representation learned by the classifier is specific to the set of classes the model was trained on.
# ->The densely connected layer no longer contain any information about where the object are located. 

# Representation extracted by specific convolution layers depends on the depth of the layer in the model 
# layers that comes earlier in the model extract generic feature maps : edges, lolor, textures 
# layers higher-up extract abstract concepts : cat ear, dog eye. 




# Part 1-Take the convolutional base of a previous trained network 

import keras 

# Instantiate the VGG16 model 
# include_top=false not include the top of the network. 
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet', 
                include_top=false,
                input_shape=(150,150,3))

conv_base.summary()
#-> the final feature map has shape (4,4,512)
#-> that the features on the top of which we stick a densely-connected classifier. 



# Part 1......and running the data throught it 
# Extract features from theses images calling the predicted methods of the conv_base model 

# import the dataset 
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# à modifier
base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# Create extract features function 
def extract_features(directory, sample_count):
    # 4, 4, 512 ->  the final feature map of conv_base has shape (4,4,512)
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    # pre-processing of the images with datagen.flow_from_directory
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        # extract the features from the conv_base with conv_base.predict 
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


# Apply extractct feature function to the training, test, validation images dataset. 
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
# shape of the extracted features (samples, 4, 4 , 512)
# -> must be flattened to (samples, 8192)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# Part 2- Train a new classifier on top of the output 
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


