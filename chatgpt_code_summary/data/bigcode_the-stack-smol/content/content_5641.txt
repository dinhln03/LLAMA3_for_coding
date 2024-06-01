import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 64, 64#572, 572
out_height, out_width = 64, 64#388, 388
GPU = False
torch.manual_seed(0)


class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()  #  necessarry?

        enc1 = []

        enc1.append(torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        enc1.append(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        enc1.append(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        enc1.append(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        enc1.append(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        enc1.append(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1))
        enc1.append(torch.nn.BatchNorm2d(32))
        enc1.append(torch.nn.ReLU())

        self.enc1 = torch.nn.Sequential(*enc1)

        self.out = torch.nn.Conv2d(32, 1, kernel_size, padding=0, stride=1)
