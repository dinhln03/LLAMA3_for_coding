import warnings

import torch
import kornia
import numpy as np


class MetricMAD:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3

class MetricBgrMAD:
    def __call__(self, pred, true):
        bgr_mask = true == 0
        return (pred[bgr_mask] - true[bgr_mask]).abs_().mean() * 1e3

class MetricFgrMAD:
    def __call__(self, pred, true):
        fgr_mask = true > 0
        return (pred[fgr_mask] - true[fgr_mask]).abs_().mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()

    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000

    def gauss_gradient(self, img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            img_filtered_x = kornia.filters.filter2D(img[None, None, :, :], self.filter_x, border_type='replicate')[0, 0]
            img_filtered_y = kornia.filters.filter2D(img[None, None, :, :], self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x ** 2 + img_filtered_y ** 2).sqrt()

    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x ** 2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma ** 2


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2
