""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from .channels import C


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, apply_sigmoid_to_output=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, C[0])
        self.down1 = Down(C[0], C[1])
        self.down2 = Down(C[1], C[2])
        self.down3 = Down(C[2], C[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(C[3], C[4] // factor)  # switch do Double CONV if stick do 8x spatial down
        self.up1 = Up(C[4], C[3] // factor, bilinear)
        self.up2 = Up(C[3], C[2] // factor, bilinear)
        self.up3 = Up(C[2], C[1] // factor, bilinear)
        self.up4 = Up(C[1], C[0], bilinear)
        self.outc = OutConv(C[0], n_classes) if apply_sigmoid_to_output is False else OutConv(C[0], n_classes, sigmoid=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
