import cv2 as cv
from PIL import Image
import numpy as np


WHITE_VALUE = 255


def crop_window(image, size, interpolator, threshold=127, shift=10) -> np.array:
    img_gr = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_gr = cv.threshold(img_gr, threshold, WHITE_VALUE, cv.THRESH_BINARY)[1]
    borders = [i for i, st in enumerate(img_gr) if 0 in st]
    b0, b1 = (borders[0] - shift, borders[-1] + shift) if len(borders) != 0 else (0, None)
    
    img = image[b0:b1, :, :]
    im_pil = Image.fromarray(img).convert('RGBA')
    img = im_pil.resize((size, size), resample=interpolator)
    return np.array(img, dtype=float).reshape((size, size, 4)) / 255


class RecognitionSample: # string for recognition: dict[(size, method): list[np.array]]

    def __init__(self, keys, alphabet_len):
        self.data = {key: [] for key in keys}
        self.alphabet_len = alphabet_len

    def put(self, key, frame: np.array):
        size = key[0]
        if not frame.shape == (size, size, 4):
            raise ValueError(f"expected ({size}, {size}, 4), got {frame.shape}")
        self.data[key].append(frame)

    def construct(self, path, step=10, window=75, threshold=127,):
        image_string = cv.imread(path)
        h, w, _ = image_string.shape
        for i in range(0, w-window, step):
            img = image_string[:, i:i+window, :]
            for size, method in self.data.keys():
                img_crop = crop_window(img, size, method)
                sample.put((size, method), img_crop)
        self.count_frames = i + 1

    def get_key_data(self, key) -> list[np.array]:
        return self.data[key]

    def get_params(self):
        return (self.count_frames, self.alphabet_len)
