import cv2 as cv
from PIL import Image
import numpy as np


WHITE_VALUE = 255


def vertical_bords(image, threshold=127, shift=10) -> (int, int): # возвращает верхнюю и нижнюю границы, по которым можно обрезать
    #удаляем белое пространство сверху и снизу (если найдётся)
    #удаляем следы соседних строк (если получится)
    
    img_gr = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_gr = cv.threshold(img_gr, threshold, WHITE_VALUE, cv.THRESH_BINARY)[1]
    white_spaces = list()
    b0, b1 = 0, image.shape[0]
    w0, w1 = -1, 0
    for i, st in enumerate(img_gr):
        if 0 in st:
            b0 = b0 if b0 else i
            b1 = i
            if w1:
                white_spaces.append((w0, w1))
        else:
            if w1 != i - 1:
                w0 = i
            w1 = i
    if w1 == i:
        white_spaces.append((w0, w1))
    w0 = white_spaces[0][1]
    w1 = white_spaces[-1][0]
    g0 = abs(max(b0, w0) - shift)
    g1 = min(b1, w1) + shift
    return g0, g1


def calc_mode(image_depth: int):
    match image_depth:
        case 1:
            return 'L'
        case 2:
            return 'LA'
        case 3:
            return 'RGB'
        case 4:
            return 'RGBA'
        case _:
            raise ValueError(f"Got unexpected value of image depth: {image_depth}")
    

def crop_window(image, size, interpolator) -> np.array:
    g0, g1 = vertical_bords(image)
    img = image[g0:g1, :, :]
    mode = calc_mode(size[-1])
    im_pil = Image.fromarray(img).convert(mode)
    pl_size = size[:2]
    img = im_pil.resize(pl_size, resample=interpolator)
    return np.array(img, dtype=float).reshape(size) / 255


class RecognitionSample: # string for recognition: dict[(size, method): list[np.array]]

    def __init__(self, keys, alphabet_len: int):
        self.data = {key: [] for key in keys}
        self.alphabet_len = alphabet_len
        self.whitespace_index = set()

    def put(self, key, frame: np.array):
        size = key[0]
        if not frame.shape == size:
            raise ValueError(f"expected {size}, got {frame.shape}")
        self.data[key].append(frame)

    def construct(self, path, step=10, window=75, threshold=127):
        image_string = cv.imread(path)
        h, w, _ = image_string.shape
        for i in range(0, w-window, step):
            img = image_string[:, i:i+window, :]
            for size, method in self.data.keys():
                img_crop = crop_window(img, size, method)
                if sum(img_crop.flatten()) == 0:
                    self.whitespace_index.add(i // step)
                self.put((size, method), img_crop)
        self.count_frames = i // step + 1

    def get_key_data(self, key) -> np.array:
        return np.array(self.data[key])

    def get_params(self):
        return (self.count_frames, self.alphabet_len)
