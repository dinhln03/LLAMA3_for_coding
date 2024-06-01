import argparse
import multiprocessing
import random
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy as np
from chainer import iterators, optimizers, serializers
from chainer.datasets import TransformDataset, get_cifar10
from chainer.training import StandardUpdater, Trainer, extensions

import augmentation
from metric_learning import MetricLearnClassifier
from modified_evaluator import ModifiedEvaluator
from modified_updater import ModifiedUpdater
from resnet import ResNet50


def apply_augmentation(inputs, mean, std, angle=(-5, 5), scale=(1, 1.2),
                       crop_size=None, train=True):
    img, label = inputs
    img = img.copy()
    img = img.transpose(1, 2, 0)

    if train:
        img, _ = augmentation.gamma_correction(img)

    img -= mean[None, None, :]
    img /= std[None, None, :]

    if train:
        img, _ = augmentation.random_rotate(img, angle=angle)
        if np.random.rand() < 0.5:
            img, _ = augmentation.mirror(img)
        if np.random.rand() < 0.5:
            img, _ = augmentation.flip(img)
        img, _ = augmentation.random_resize(img, scale=scale)
    if crop_size is not None:
        rnd1 = np.random.randint(img.shape[0] - crop_size)
        rnd2 = np.random.randint(img.shape[1] - crop_size)
        img = img[rnd1:rnd1 + crop_size, rnd2:rnd2 + crop_size, :]

    img = img.transpose(2, 0, 1)

    return img, label


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--report_trigger', '-rt', type=str, default='1e',
                        help='Interval for reporting(Ex.100i, default:1e)')
    parser.add_argument('--save_trigger', '-st', type=str, default='1e',
                        help='Interval for saving the model(Ex.100i, default:1e)')
    parser.add_argument('--load_model', '-lm', type=str, default=None,
                        help='Path of the model object to load')
    parser.add_argument('--load_optimizer', '-lo', type=str, default=None,
                        help='Path of the optimizer object to load')
    args = parser.parse_args()

    start_time = datetime.now()
    save_dir = Path('output/{}'.format(start_time.strftime('%Y%m%d_%H%M')))

    random.seed(args.seed)
    np.random.seed(args.seed)
    cupy.random.seed(args.seed)

    model = MetricLearnClassifier(ResNet50(), 512, 10,
                                  method='arcface', final_margin=0.5,
                                  final_scale=64, target_epoch=100)

    if args.load_model is not None:
        serializers.load_npz(args.load_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=1e-3, weight_decay_rate=5e-4, amsgrad=True)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    if args.load_optimizer is not None:
        serializers.load_npz(args.load_optimizer, optimizer)

    train_data, valid_data = get_cifar10(scale=255.)
    mean = np.mean([x for x, _ in train_data], axis=(0, 2, 3))
    std = np.std([x for x, _ in train_data], axis=(0, 2, 3))

    train_transform = partial(apply_augmentation, mean=mean, std=std, crop_size=28, train=True)
    valid_transform = partial(apply_augmentation, mean=mean, std=std, crop_size=28, train=True)

    train_data = TransformDataset(train_data, train_transform)
    valid_data = TransformDataset(valid_data, valid_transform)

    train_iter = iterators.SerialIterator(train_data, args.batchsize)
    valid_iter = iterators.SerialIterator(valid_data, args.batchsize, repeat=False, shuffle=False)

    updater = ModifiedUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=save_dir)

    report_trigger = (int(args.report_trigger[:-1]), 'iteration' if args.report_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.LogReport(trigger=report_trigger))
    trainer.extend(ModifiedEvaluator(valid_iter, model, device=args.gpu), name='val', trigger=report_trigger)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss',
                                           'val/main/accuracy', 'elapsed_time']), trigger=report_trigger)
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key=report_trigger[1],
                                         marker='.', file_name='loss.png', trigger=report_trigger))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key=report_trigger[1],
                                         marker='.', file_name='accuracy.png', trigger=report_trigger))

    save_trigger = (int(args.save_trigger[:-1]), 'iteration' if args.save_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.snapshot_object(model, filename='model_{0}-{{.updater.{0}}}.npz'
                                              .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_{0}-{{.updater.{0}}}.npz'
                                              .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(30, 'epoch'))

    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    (save_dir / 'training_details').mkdir()

    # Write parameters text
    with open(save_dir / 'training_details/train_params.txt', 'w') as f:
        f.write('model: {}\n'.format(model.predictor.__class__.__name__))
        f.write('n_epoch: {}\n'.format(args.epoch))
        f.write('batch_size: {}\n'.format(args.batchsize))
        f.write('n_data_train: {}\n'.format(len(train_data)))
        f.write('n_data_val: {}\n'.format(len(valid_data)))
        f.write('seed: {}\n'.format(args.seed))

    trainer.run()


if __name__ == '__main__':
    main()
