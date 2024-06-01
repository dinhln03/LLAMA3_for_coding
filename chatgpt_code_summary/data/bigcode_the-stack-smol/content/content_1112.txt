from abc import ABC, abstractmethod
import numpy as np
from .constants import EPSILON
import torch


class Loss(ABC):
    def __init__(self, expected_output, predict_output):
        self._expected_output = expected_output
        self._predict_output = predict_output

    @abstractmethod
    def get_loss(self):
        pass


def crossEntropy(expected_output, predict_output):
    return -(expected_output * torch.log(predict_output) +
             (1-expected_output) * torch.log(1-predict_output+EPSILON)).mean()


def l2(expected_output, predict_output):
    return ((predict_output - expected_output) ** 2).mean()
