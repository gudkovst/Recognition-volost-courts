import glob
import os.path

import cv2 as cv
import numpy as np


PAPER_VALUE = [176, 183, 187]
MEAN_PAPER_VALUE = sum(PAPER_VALUE) / len(PAPER_VALUE)


def vertical_borders(image, threshold=127, shift=10, min_split_width: int = 16, part_of_height: float = 0.6) -> (int, int):
    # возвращает верхнюю и нижнюю границы, по которым можно обрезать
    # удаляем белое пространство сверху и снизу (если найдётся)
    # удаляем следы соседних строк (если получится)
    height, width, _ = image.shape
    if width < min_split_width:
        return 0, height
    img_gr = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_gr = cv.threshold(img_gr, threshold, MEAN_PAPER_VALUE, cv.THRESH_BINARY)[1]
    white_spaces = list()
    b0, b1 = 0, height
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
    if w1 == len(img_gr) - 1:
        white_spaces.append((w0, w1))
    if len(white_spaces) == 0:
        g1_min, g1_max = vertical_borders(image[:, :width // 2, :])
        g2_min, g2_max = vertical_borders(image[:, width // 2:, :])
        w0 = max(g1_min, g2_min)
        w1 = min(g1_max, g2_max)
    else:
        w0 = white_spaces[0][1]
        w1 = white_spaces[-1][0]
    g0 = abs(max(b0, w0) - shift)
    g1 = (b1 if w1 < height * part_of_height else min(b1, w1)) + shift
    return g0, g1


def fill_borders(image, bbox, filled_value):
    """
    Заполняет изображение вокруг границ
    :param image:
    :param bbox: bounding box of needed text in format (x_min, y_min, x_max, y_max)
    :param filled_value:
    :return: None
    """
    assert len(bbox) == 4
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return

    for j in range(x_min, x_max):
        for i in range(0, y_min):
            image[i][j] = filled_value
        for i in range(y_max, image.shape[0]):
            image[i][j] = filled_value


def handler_line(line_image_path, save_path, step: int):
    line_image = cv.imread(line_image_path)
    h, w, _ = line_image.shape
    filled_value = np.mean(line_image, axis=(0, 1)) + 10
    for i in range(0, w - step, step):
        image_fragment = line_image[:, i:i+step, :]
        y_min, y_max = vertical_borders(image_fragment)


        bbox = (i, y_min, i + step, y_max)
        #filled_value = np.mean(image_fragment, axis=(0, 1))
        fill_borders(line_image, bbox, filled_value)
    cv.imwrite(save_path, line_image)


if __name__ == "__main__":
    path = r"C:\Users\gudko\history_envs\easyocr_env\big_block"
    for i, im in enumerate(glob.glob(os.path.join(path, '*.jpg'))):
        im_path = os.path.join(path, im)
        save_path = fr"C:\Users\gudko\history_envs\easyocr_env\big_block\line_{i + 100}_roi.jpg"
        handler_line(im_path, save_path, 32)
