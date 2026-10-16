import cv2 as cv
import numpy as np


def scale_to_h(img, target_height):
    h, w = img.shape[:2]

    if h == target_height:
        return img

    if h == 0 or img.size == 0:
        return np.zeros(
            (target_height, w) + img.shape[2:],
            dtype=img.dtype
        )

    scale = target_height / h
    target_width = max(round(scale * w), 1)

    interpolation = (
        cv.INTER_AREA if scale <= 1
        else cv.INTER_LINEAR
    )

    return cv.resize(
        img,
        (target_width, target_height),
        interpolation=interpolation
    )


def measure(line, smoothness=1.0, extra=0.3, range_factor=4):
    h, w = line.shape

    smoothed = cv.GaussianBlur(
        line,
        (0, 0),
        sigmaX=h * smoothness,
        sigmaY=h * 0.5,
        borderType=cv.BORDER_CONSTANT
    )

    smoothed += (
        0.001
        * cv.blur(
            smoothed,
            (w, int(h * 0.5)),
            borderType=cv.BORDER_CONSTANT
        )
    )

    a = np.argmax(
        smoothed,
        axis=0
    ).astype(np.uint16)

    kernel = cv.getGaussianKernel(
        int(8.0 * h * extra + 1),
        h * extra
    )

    center = cv.filter2D(
        a,
        cv.CV_16U,
        kernel,
        borderType=cv.BORDER_REFLECT
    ).flatten()

    deltas = abs(
        np.arange(h)[:, None]
        - center[None, :]
    )

    mad = np.mean(deltas[line != 0])

    r = int(1 + range_factor * mad)

    return center, r


def center_normalize(img, target_height=48):
    intermediate_height = int(target_height * 1.5)

    if intermediate_height < img.shape[0]:
        img = scale_to_h(
            img,
            intermediate_height
        )

    cval = int(np.max(img))

    temp = (
        img.astype(np.float32) / 255
    )

    temp = np.max(temp) - temp

    amax = np.max(temp)

    if amax == 0:
        return scale_to_h(
            np.zeros_like(img),
            target_height
        )

    inverted = temp / amax

    center, r = measure(inverted)

    padded = cv.copyMakeBorder(
        img,
        r,
        r,
        0,
        0,
        cv.BORDER_CONSTANT,
        value=cval
    )

    center = center + r - r
    new_h = 2 * r

    dewarped = [
        padded[c:c + new_h, i]
        for i, c in enumerate(center)
    ]

    dewarped = np.swapaxes(
        np.array(dewarped, dtype=np.uint8),
        1,
        0
    )

    return scale_to_h(
        dewarped,
        target_height
    )


def final_preparation(img):
    data = img.astype(np.float32) / 255

    data = np.expand_dims(
        data,
        axis=-1
    )

    amax = np.max(data)

    if amax > 0:
        data = data / amax

    data = np.max(data) - data

    data = np.swapaxes(
        data,
        1,
        0
    )

    data = np.vstack([
        np.zeros(
            (16, data.shape[1], 1)
        ),
        data,
        np.zeros(
            (16, data.shape[1], 1)
        )
    ])

    data = (data * 255).astype(np.uint8)

    data = np.squeeze(
        data,
        axis=-1
    )

    return data


def prepare_image(image_path):
    img = cv.imread(
        str(image_path),
        cv.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(image_path)

    normalized = center_normalize(img)

    prepared = final_preparation(normalized)

    return prepared