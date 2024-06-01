import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_trainable(net, val):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val


def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses['g'], label='generative loss')
    plt.plot(losses['d'], label='discriminitive loss')
    plt.legend()
    plt.show()


def render_bboxes(bboxes_batch, labels_batch, shape):
    renders = []

    for i in range(len(bboxes_batch)):
        bboxes = bboxes_batch[i]
        labels = labels_batch[i]
        canvas = np.zeros(shape, dtype=np.float32)
        canvas += 255

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            top, left, bottom, right = bbox
            label = labels[j]
            color = (np.where(label==1)[0][0] + 1) * 10
            canvas[top:bottom, left:right, 0] = color

        canvas /= 255
        renders.append(canvas)

    return np.array(renders)


def save_batch(images, epoch, path, suffix=''):
    samples_path = os.path.join(path, 'samples')
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    num_images = images.shape[0]
    num_rows = images.shape[1]
    num_cols = images.shape[2]

    canvas = np.zeros((num_rows, num_images * num_cols, 1), dtype=images.dtype)
    for i in range(num_images):
        canvas[0:num_rows, i * num_cols:(i + 1) * num_cols] = images[i]

    img = canvas
    img *= 255
    img = Image.fromarray(np.squeeze(img))
    img = img.convert('L')
    img.save(samples_path + f'/{epoch}_{suffix}.png')


def load_model(model, path, name):
    model_path = os.path.join(path, name + '.h5')
    model.load_weights(model_path)


def save_model(model, path, name):
    model_path = os.path.join(path, name + '.h5')
    model.save_weights(model_path)
