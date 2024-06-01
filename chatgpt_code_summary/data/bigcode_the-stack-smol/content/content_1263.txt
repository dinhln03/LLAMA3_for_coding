from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.layers import Dropout, TimeDistributed
try:
    from tensorflow.python.keras.layers import CuDNNLSTM as lstm
except:
    from tensorflow.keras.layers import Dense, Activation, LSTM as lstm
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model as lm

import numpy as np
import random
import sys
import io
from midi import Midi


class Model:
    def create(self, size, unique_notes, optimizer=None, hidden_size=128):
        self.model = Sequential()
        self.model.add(lstm(hidden_size, input_shape=(
            size, unique_notes), return_sequences=True))
        self.model.add(lstm(hidden_size))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(unique_notes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(
            lr=0.01) if optimizer == None else optimizer)

    def load_from_file(self, name="model.h5"):
        self.model = lm(name)

    def save_to_file(self, name="model.h5"):
        self.model.save(name)

    def learn(self, inputs, outputs, batch_size=256, epochs=185):
        self.model.fit(inputs, outputs,
                       batch_size=batch_size,
                       epochs=epochs, verbose=True)

    def predict(self, arr):
        return self.model.predict(arr)
