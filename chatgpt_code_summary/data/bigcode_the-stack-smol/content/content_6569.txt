import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = F.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        score = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = F.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + 0.5 * dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class DSVLovaszHingeLoss(nn.Module):
    def __init__(self):
        super(DSVLovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        for i in range(target.shape[0]):
            if not torch.sum(target[i]).data.cpu().numpy() > 1:
                target[i] = -1

        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True, ignore=-1)

        return loss
