import keras
from sklearn.metrics import roc_auc_score
from src.predictionAlgorithms.machineLearning.helpers.validation import Validation
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

class Callbacks(keras.callbacks.Callback):
    validationSequences = []
    algorithm = None
    number = 1
    validation_frequency = 1
    size = 64
    step = 1
    base = 4

    def set_step(self, step):
        self.step = step
        return self

    def set_base(self, base):
        self.base = base
        return base

    def set_size(self, size):
        self.size = size
        return self

    def set_validation_frequency(self, frequency):
        self.validation_frequency = frequency
        return self

    def set_validation_data(self, validation_data):
        self.validationSequences = validation_data
        return self

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        return self

    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        epoch_graphs = glob.glob('../output/*')
        for f in epoch_graphs:
            os.remove(f)


    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if self.number % self.validation_frequency != 0:
            self.number += 1
            return
        validation = Validation()
        validation.set_validation_data(self.validationSequences)\
            .set_dimensions(self.size)\
            .set_base(self.base)\
            .set_step(self.step)\
            .validate(self.algorithm)

        self.number += 1

        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig('../output/Epoch-{}.png'.format(epoch))
            plt.close()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
