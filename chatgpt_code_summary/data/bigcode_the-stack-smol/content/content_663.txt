import pandas as pd
import numpy as np
import wave
from scipy.io import wavfile
import os
import librosa
import pydub
import ffmpeg
from librosa.feature import melspectrogram
import warnings
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from PIL import Image
import sklearn

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN, Conv1D, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from keras.models import load_model

import boto3
import botocore

def model_input():
    # Load the trained model
    model = load_model("best_model.h5")

    #Access S3 Bucket and Download the audio file
    BUCKET_NAME = 'thunderstruck-duck' # replace with your bucket name
    KEY = "sample_mp3.mp3" # replace with your object key

    s3 = boto3.client('s3',
                        aws_access_key_id='AKIAISITTOGCJRNF46HQ',
                        aws_secret_access_key= 'bq/VRAme7BxDMqf3hgEMLZdrJNVvrtdQ4VmoGAdB',
                        )
    BUCKET_NAME = "thunderstruck-duck"


    try:
        s3.download_file(BUCKET_NAME, KEY, "sample_mp3.mp3")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
    # else:
    #     raise
    
    #Load the audio data using librosa


    wave_data, wave_rate = librosa.load("sample_mp3.mp3")
    wave_data, _ = librosa.effects.trim(wave_data)
    #only take 5s samples and add them to the dataframe
    song_sample = []
    sample_length = 5*wave_rate
    #The variable below is chosen mainly to create a 216x216 image
    N_mels=216
    for idx in range(0,len(wave_data),sample_length): 
        song_sample = wave_data[idx:idx+sample_length]
        if len(song_sample)>=sample_length:
            mel = melspectrogram(song_sample, n_mels=N_mels)
            db = librosa.power_to_db(mel)
            normalised_db = sklearn.preprocessing.minmax_scale(db)
            filename = "sample_mel.tif"
            db_array = (np.asarray(normalised_db)*255).astype(np.uint8)
            db_image =  Image.fromarray(np.array([db_array, db_array, db_array]).T)
            db_image.save("{}{}".format("upload_mel/",filename))
    
    #Create a DF that will take the created Melspectogram directory
    data_df = pd.DataFrame([{'bird': "sample bird", 'song_sample': f"/app/upload_mel/{filename}"}])
    
    # Users/HyunsooKim/Desktop/Boot_Camp/Homework/BIRD_CALL/upload_mel/{filename}"}])
    
    #Compile the model
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    model.compile(loss="categorical_crossentropy", optimizer='adam')

    #Since we only have 1 melspectogram passing into the model, set batch size to 1 and the size of that image so the model can take the image file.
    validation_batch_size_full = 1
    target_size = (216,216)

    train_datagen_full = ImageDataGenerator(
        rescale=1. / 255
    )
    #Pass the columns into the model
    validation_datagen_full = ImageDataGenerator(rescale=1. / 255)
    validation_generator_full = validation_datagen_full.flow_from_dataframe(
        dataframe = data_df,
        x_col='song_sample',
        y_col='bird',
        directory='/',
        target_size=target_size,
        shuffle=False,
        batch_size=validation_batch_size_full,
        class_mode='categorical')
    
    #Run the model
    preds = model.predict_generator(validation_generator_full)

    #We want to find the "INDEX" of maximum value within the pred, a numpy array. Use np.argmax and index into 0th element.
    result = np.argmax(preds[0])

    #load in the index dataframe, so we can find the name of the bird that matches the index of our result
    index_df = pd.read_csv('xeno-canto_ca-nv_index.csv')
    #rename the english_cname to birds for better access and clearity
    bird_list = pd.DataFrame(index_df.english_cname.unique())
    bird_list.columns = ["birds"]

    #We are almost done. Save the percentage and the name of the bird into a variable and print it out!
    percentage = preds[0][result]
    Name_of_bird = bird_list['birds'][result]

    print(f"This bird is {percentage} likely {Name_of_bird}")


    final_data = {"likelihood": percentage, "name_of_bird": Name_of_bird}

    return final_data

    if __name__ == "__main__":
        print(model_input())