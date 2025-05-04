import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


GRAY_VALUE = 127
WHITE_VALUE = 255


def show(img):
    plt.imshow(img)


def fill_fields(image_ref):    
    def __round__(index, direction):
        for j, p in enumerate(image_ref[index]):
            if p == 0:
                image_ref[index][j] = WHITE_VALUE
                i = index + direction
                while image_ref[i][j] == 0:
                    image_ref[i][j] = WHITE_VALUE
                    i += direction
                    
    __round__(0, 1)
    __round__(image_ref.shape[0] - 1, -1)


def repaint_columns(image_ref, coef=0.99, threshold=25, paint=False, verbose=False):
    deleted = []
    counter, white_col_begin, white_col_end = 0, -1, 0
    for col in range(image_ref.shape[1]):
        count_white = sum([st[col] for st in image_ref]) / WHITE_VALUE
        if count_white > image_ref.shape[0] * coef:
            counter += 1
            if white_col_end != col - 1:
                white_col_begin = col
            white_col_end = col
            if verbose:
                print('col: ', col, count_white)
            if paint:
                for i in range(image_ref.shape[0]):
                    image_ref[i][col] = GRAY_VALUE
        else:
            if counter > threshold:
                deleted.append((white_col_begin, white_col_end))
            counter = 0
    if counter > threshold:
        deleted.append((white_col_begin, white_col_end))
    return deleted


def repaint_strings(image_ref, coef=0.8, threshold=25, paint=False, verbose=False):
    deleted = []
    counter, white_str_begin, white_str_end = 0, -1, 0
    for i, st in enumerate(image_ref):
        count_white = sum(st) / WHITE_VALUE
        if count_white > image_ref.shape[1] * coef:
            counter += 1
            if white_str_end != i - 1:
                white_str_begin = i
            white_str_end = i
            if verbose:
                print('str: ', i, count_white)
            if paint:
                for j in range(image_ref.shape[1]):
                    image_ref[i][j] = GRAY_VALUE
        else:
            if counter > threshold:
                deleted.append((white_str_begin, white_str_end))
            counter = 0
    if counter > threshold:
        deleted.append((white_str_begin, white_str_end))
    return deleted


def cut_strings(img, deleted):
    imgs = []
    d = [(0, 0)] + deleted + [img.shape]
    for i in range(len(d) - 1):
        imgs.append(img[d[i][1]:d[i+1][0]])
    return imgs


def cut_columns(img, deleted):
    d = [(0, 0)] + deleted + [img.shape]
    imgs = [[] for i in d[1:]]
    for st in img:
        for i in range(len(d) - 1):
            imgs[i].append(st[d[i][1]:d[i+1][0]])
    return [np.array(im) for im in imgs]


def get_binary_image(image_path, threshold=127):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, threshold, WHITE_VALUE, cv.THRESH_BINARY)
    img_tr = img[1]
    fill_fields(img_tr)
    return img_tr
