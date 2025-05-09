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
                while 0 <= i < image_ref.shape[0] and image_ref[i][j] == 0:
                    image_ref[i][j] = WHITE_VALUE
                    i += direction
                    
    __round__(0, 1)
    __round__(image_ref.shape[0] - 1, -1)


def get_column(img, num):
    if not 0 <= num < img.shape[1]:
        raise IndexError('Number of column is invalid')
    return [st[num] for st in img]
    

def merge_deleted(deleted, threshold):
    merged_deleted = []
    merged_begin = None
    for d1, d2 in zip(deleted[:-1], deleted[1:]):
        if d2[0] - d1[1] > threshold:
            if merged_begin is None:
                merged_deleted.append(d1)
            else:
                merged_deleted.append((merged_begin, d1[1]))
                merged_begin = None
        else:
            if merged_begin is None:
                merged_begin = d1[0]
    if merged_begin is not None:
        merged_deleted.append((merged_begin, deleted[-1][1]))
    return merged_deleted


def repaint(image_ref, direction, coef=0.8, threshold=25, merge_threshold=25, paint=False, verbose=False): # direction = 0 if str = 1 if col
    def __paint__():
        if direction:
            for i in range(image_ref.shape[0]):
                    image_ref[i][num_line] = GRAY_VALUE
        else:
            for j in range(image_ref.shape[1]):
                    image_ref[num_line][j] = GRAY_VALUE
                
    
    deleted = [(0, 0)]
    counter, white_begin, white_end = 0, -1, 0
    for num_line in range(image_ref.shape[direction]):
        line = get_column(image_ref, num_line) if direction else image_ref[num_line]
        count_white = sum(line) / WHITE_VALUE
        if count_white > image_ref.shape[1 - direction] * coef:
            counter += 1
            if white_end != num_line - 1:
                white_begin = num_line
            white_end = num_line
            if verbose:
                title = 'col: ' if direction else 'str: '
                print(title, num_line, count_white)
            if paint:
                __paint__()
        else:
            if counter > threshold:
                deleted.append((white_begin, white_end))
            counter = 0
    if counter > threshold:
        deleted.append((white_begin, white_end))
    deleted.append((image_ref.shape[direction], image_ref.shape[direction]))
    return merge_deleted(deleted, merge_threshold)


def repaint_columns(image_ref, coef=0.99, threshold=25, merge_threshold=25, paint=False, verbose=False):
    return repaint(image_ref, 1, coef, threshold, merge_threshold, paint, verbose)


def repaint_strings(image_ref, coef=0.8, threshold=25, merge_threshold=25, paint=False, verbose=False):
    return repaint(image_ref, 0, coef, threshold, merge_threshold, paint, verbose)


def cut_strings(img, deleted):
    imgs = []
    d = deleted + []
    if d[0][0] != 0:
        d = [(0, 0)] + d
    if d[-1][1] != img.shape[0]:
        d += [img.shape]
    for i in range(len(d) - 1):
        imgs.append(img[d[i][1]:d[i+1][0]])
    return imgs


def cut_columns(img, deleted):
    d = deleted + []
    if d[0][0] != 0:
        d = [(0, 0)] + d
    if d[-1][1] != img.shape[1]:
        d += [(img.shape[1], img.shape[1])]
    imgs = [[] for i in d[1:]]
    for st in img:
        for i in range(len(d) - 1):
            imgs[i].append(st[d[i][1]:d[i+1][0]])
    return [np.array(im) for im in imgs]


def graphic_columns_whites(img):
    whites = []
    cols = range(img.shape[1])
    for col in cols:
        whites.append(sum(get_column(img, col)) / WHITE_VALUE)
    plt.plot(cols, whites, 'b')
    return list(zip(cols, whites))


def get_binary_image(image_path, threshold=127):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, threshold, WHITE_VALUE, cv.THRESH_BINARY)
    img_tr = img[1]
    fill_fields(img_tr)
    return img_tr
