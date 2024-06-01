import os

import numpy as np
from keras import backend as K
from keras.losses import mean_absolute_error

import utils
from model import wdsr_b


def psnr(hr, sr, max_val=2):
    mse = K.mean(K.square(hr - sr))
    return 10.0 / np.log(10) * K.log(max_val ** 2 / mse)


def data_generator(path, batch_size=8, input_shape=96, scale=2):
    '''data generator for fit_generator'''
    fns = os.listdir(path)
    n = len(fns)
    i = 0
    while True:
        lrs, hrs = [], []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(fns)
            fn = fns[i]
            fn = os.path.join(path, fn)
            lr, hr = utils.pair(fn, input_shape, scale)
            lr = utils.normalization(lr)
            hr = utils.normalization(hr)
            lrs.append(lr)
            hrs.append(hr)
            i = (i + 1) % n
        lrs = np.array(lrs)
        hrs = np.array(hrs)
        yield lrs, hrs


model = wdsr_b()
model.compile(optimizer='adam',
              loss=mean_absolute_error, metrics=[psnr])
model.fit_generator(data_generator('./datasets/train/'),
                    steps_per_epoch=50,
                    epochs=1250)
