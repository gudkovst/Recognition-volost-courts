import json
import numpy as np

from detect_string import process_handwritten_book_page


def area(bbox):
    assert len(bbox) == 4
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)


def IoU_single(true_bbox, pred_bbox):
    assert len(true_bbox) == len(pred_bbox) == 4
    true_x_min, true_y_min, true_x_max, true_y_max = true_bbox
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_bbox

    intersect_x_min = max(true_x_min, pred_x_min)
    intersect_x_max = min(true_x_max, pred_x_max)
    intersect_y_min = max(true_y_min, pred_y_min)
    intersect_y_max = min(true_y_max, pred_y_max)

    intersect_area = area((intersect_x_min, intersect_y_min, intersect_x_max, intersect_y_max))
    union_area = area(true_bbox) + area(pred_bbox) - intersect_area
    return max(0, intersect_area / (union_area + np.finfo(float).eps))


def IoU_list(true_bboxes, pred_bboxes):
    return [max([IoU_single(true_bbox, pred_bbox) for pred_bbox in predicted_bboxes]) for true_bbox in true_bboxes]


def calculate_average_IoU(true_bboxes, pred_bboxes):
    res = IoU_list(true_bboxes, pred_bboxes)
    return sum(res) / len(res)


def y_center(bbox):
    return bbox[1] + bbox[3] / 2


def load_true_bboxes(path):
    with open(path, 'r') as file:
        data = json.load(file)
    res = list()
    for annot in data["annotations"]:
        x_min, y_min, width, height = annot["bbox"]
        x_min -= data["shift"]["x"]
        y_min -= data["shift"]["y"]
        bbox = tuple(map(int, (x_min, y_min, x_min + width, y_min + height)))
        res.append(bbox)
    return sorted(res, key=y_center)


def get_easyOCR_bboxes(image_path):
    _, lines = process_handwritten_book_page(image_path)
    return sorted(lines, key=y_center)


if __name__ == "__main__":
    bboxes_path = r"C:\Users\gudko\history_envs\easyocr_env\true_labels.json"
    image_path = r"C:\Users\gudko\history_envs\ultralytics_env\21-03-project\dop_pages\aa0feaf7-81dd258a-IMG_20230824_133627\97 block.jpg"
    predicted_bboxes = get_easyOCR_bboxes(image_path)

    true_bboxes = load_true_bboxes(bboxes_path)
    print(f"Count of true {len(true_bboxes)} and predicted {len(predicted_bboxes)} bboxes")
    print(calculate_average_IoU(true_bboxes, predicted_bboxes))