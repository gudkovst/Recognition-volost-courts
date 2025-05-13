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


def convert_image(image, size: int = 10, interpolator=Image.BICUBIC) -> np.array:
    b = image.getbbox()
    img = image.crop(b)
    img = img.resize((size, size), resample=interpolator)
    #img = image.convert('LA')
    img = [np.array([c[0], c[0], c[0], c[1]]) for c in img.getdata()] if image.mode != 'RGBA' else img
    #img = [(c[0], c[1]) for c in image.getdata()]
    #L = R * 299/1000 + G * 587/1000 + B * 114/1000
    #img = [(c[0] * 1000 / (3*299), c[0]*1000 / (3*587), c[0]*1000 / (3*114), c[1]) for c in image.getdata()] if image.mode != 'RGBA' else image
    return np.array(img, dtype=float).reshape((size, size, 4)) / 255


def load_images(path: str, flatten: bool = True, size: int = 10, interpolator=Image.BICUBIC) -> (np.array, np.array):
    images = []
    labels = []
    for pict in os.listdir(path):
        with Image.open(os.path.join(path, pict)) as image:
            img = convert_image(image, size, interpolator)
            if flatten:
                img = img.flatten()
            images.append(img)
            labels.append(get_liter(pict))
    return np.array(images), np.array(labels)


def load_distributed_images(root_dir, size: int, interpolator=Image.BICUBIC, flatten: bool = False):
    images = []
    labels = []
    for liter in os.listdir(root_dir):
        liter_dir = os.path.join(root_dir, liter)
        labels += [get_liter_dir(liter)] * len(os.listdir(liter_dir))
        imgs, _ = load_images(liter_dir, flatten, size, interpolator)
        images.extend(imgs)
    return np.array(images), np.array(labels)
