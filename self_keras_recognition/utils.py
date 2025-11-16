import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn import metrics


def get_liter(name: str) -> str:
    liter = name.split('-')[0].split()[0]
    return '\U00000462' if liter == "ять" else liter.lower()


def get_liter_dir(name: str) -> str:
    unique_liters = {"д верхнее": 'd', "л длинное": 'l', "т молотком": 't'}
    if name.lower() in unique_liters:
        return unique_liters[name.lower()]
    liter = name.split()[0]
    if 'большая' in name:
        return liter.upper()
    if liter == 'ять':
        return '\U00000462'.lower()
    return liter.lower()


def fill_fields(image_ref):
    WHITE_VALUE = 255
    
    def __round__(index, direction):
        for j, p in enumerate(image_ref[index]):
            if sum(p) == 0:
                image_ref[index][j] = WHITE_VALUE
                i = index + direction
                while 0 <= i < image_ref.shape[0] and sum(image_ref[i][j]) == 0:
                    image_ref[i][j] = WHITE_VALUE
                    i += direction
                    
    __round__(0, 1)
    __round__(image_ref.shape[0] - 1, -1)


def convert_image(image, **kwargs) -> np.array:
    size = kwargs.get('size', 10)
    interpolator = kwargs.get('interpolator', Image.BICUBIC)
    mode = kwargs.get('mode', 'RGBA')
    b = image.getbbox()
    img = image.crop(b)
    img = img.convert(mode)
    img = img.resize((size, size), resample=interpolator)
    return np.array(img, dtype=float).reshape((size, size, len(mode))) / 255


def load_images(path: str, **kwargs) -> (np.array, np.array):
    images = []
    labels = []
    for pict in os.listdir(path):
        with Image.open(os.path.join(path, pict)) as image:
            img = convert_image(image, **kwargs)
            if kwargs.get('flatten', False):
                img = img.flatten()
            images.append(img)
            labels.append(get_liter(pict))
    return np.array(images), np.array(labels)


def load_distributed_images(root_dir, **kwargs):
    images = []
    labels = []
    for liter in os.listdir(root_dir):
        liter_dir = os.path.join(root_dir, liter)
        labels += [get_liter_dir(liter)] * len(os.listdir(liter_dir))
        imgs, _ = load_images(liter_dir, **kwargs)
        images.extend(imgs)
    return np.array(images), np.array(labels)
