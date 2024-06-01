"""Utilities for generating synthetic segmentation datasets."""

import os
from typing import Tuple
from pathlib import Path

import numpy as np
from skimage.draw import random_shapes
from skimage.transform import rotate
from skimage.io import imsave


def gen_shape_image(im_size: Tuple[int, int], max_shapes: int=10, overlap: bool=False, rotation: bool=False):
    # Generate an image with random shapes
    img, shapes = random_shapes(im_size, max_shapes, min_size=25, max_size=150,
                                multichannel=False, allow_overlap=overlap)

    # Find each shape and get the corresponding pixels for the label map
    labels = np.zeros(im_size)
    shape_map = {'circle': 1, 'rectangle': 2, 'triangle': 3}
    for shape, coords in shapes:
        rr, cc = coords
        shape_img = img[rr[0]:rr[1], cc[0]:cc[1]]
        colors = np.bincount(shape_img.ravel()).argsort()
        shape_color = colors[-1] if colors[-1] != 255 else colors[-2]
        shape_rr, shape_cc = np.where(shape_img == shape_color)
        shape_rr += rr[0]
        shape_cc += cc[0]
        labels[shape_rr, shape_cc] = shape_map[shape]

    # If we're rotating pick a random number between -180 and 180 and then rotate
    if rotation:
        angle = np.random.uniform(-180, 180)
        img = rotate(img, angle, preserve_range=True, resize=True, cval=255).astype(np.int)
        labels = rotate(labels, angle, preserve_range=True, resize=True).astype(np.int)

    # Swap the background color to a random color to make things interesting
    background = 255
    while background in np.unique(img):
        background = np.random.randint(0, 255)
    img[img == 255] = background

    return img.astype(np.int), labels.astype(np.int)


def generate_synthetic_dataset(path, num_samples: int, im_size=(256, 256),
                               max_shapes: int=10, overlap: bool=False, p_rotate: float=0):
    path = Path(path)
    img_path = path / 'images'
    label_path = path / 'labels'
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    for i in range(num_samples):
        rotation = bool(np.random.rand(1) < p_rotate)
        img, labels = gen_shape_image(im_size, max_shapes, overlap, rotation)
        img_name = f'{i}.png'
        imsave(img_path / img_name, img)
        imsave(label_path / img_name, labels)
