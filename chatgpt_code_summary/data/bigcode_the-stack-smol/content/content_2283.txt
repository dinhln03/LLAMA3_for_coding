import json

import numpy as np
from PIL import Image


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def save_json(obj, f, ensure_ascii=True, indent=None):
    with open(f, 'w') as fp:
        json.dump(obj, fp, ensure_ascii=ensure_ascii, indent=indent)


def load_image(f, mode='RGB'):
    with Image.open(f) as image:
        return np.array(image.convert(mode))
