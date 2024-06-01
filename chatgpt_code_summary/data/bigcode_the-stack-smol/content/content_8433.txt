# Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_model():
    # Read data set
    spotify_df = pd.read_csv("spotify_data_v4.csv")

    # Extract the necessary columns we need for machine learning model
    spotify_df_clean = spotify_df[[
        'genre', 'genre_label', 'loudness', 'energy', 
        'danceability', 'instrumentalness'
    ]]

    # Assign X (data) and y (target)
    X = spotify_df_clean.drop(["genre", "genre_label"], axis=1)
    y = spotify_df_clean["genre_label"]

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Scale the data using MinMaxScaler
    # Create a MinMaxScaler model and fit it to the training data
    X_scaler = MinMaxScaler().fit(X_train)

    # Transform the training and testing data using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def scale_input(score_list):
    # Read data set
    spotify_df = pd.read_csv("spotify_data_v4.csv")

    # Extract the necessary columns we need for machine learning model
    spotify_df_clean = spotify_df[[
        'genre', 'genre_label', 'loudness', 'energy', 
        'danceability', 'instrumentalness'
    ]]

    # Assign X (data) and y (target)
    X = spotify_df_clean.drop(["genre", "genre_label"], axis=1)
    y = spotify_df_clean["genre_label"]

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Scale the data using MinMaxScaler
    # Create a MinMaxScaler model and fit it to the training data
    X_scaler = MinMaxScaler().fit(X_train)

    # Need to scale and transform the input using X_scaler which the scaler we used while training the data
    score_list_scaled = X_scaler.transform([score_list])

    return score_list_scaled