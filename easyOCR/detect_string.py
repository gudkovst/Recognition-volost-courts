import os.path

import numpy as np
import cv2
import easyocr
from typing import List, Tuple, Dict, Any
import itertools

from sympy import intervals


def transform_bboxes(
        horizontal_boxes: List[List[int]],
        free_boxes: List[List[List[int]]]
) -> list[dict[str, Any]]:
    """
        Преобразует все детекции в единый формат.

        Args:
            horizontal_boxes: Список горизонтальных боксов в формате [[x_min, x_max, y_min, y_max], ...]
            free_boxes: Список полигонов в формате [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]

        Returns:
             Отсортированный по center_y список информации о боксах слов
             в формате {'bbox': , 'center_y': , 'height': , 'type': 'horizontal' or 'free'}
        """
    all_detections = []

    # Обрабатываем горизонтальные боксы
    for hbox in horizontal_boxes:
        if len(hbox) >= 4:
            x_min, x_max, y_min, y_max = hbox[:4]
            bbox = [x_min, y_min, x_max, y_max]
            center_y = (y_min + y_max) / 2
            height = y_max - y_min
            all_detections.append({
                'bbox': bbox,
                'center_y': center_y,
                'height': height,
                'type': 'horizontal'
            })

    # Обрабатываем свободные полигоны
    for polygon in free_boxes:
        if len(polygon) == 4:
            points = np.array(polygon)
            x_min = points[:, 0].min()
            x_max = points[:, 0].max()
            y_min = points[:, 1].min()
            y_max = points[:, 1].max()
            bbox = list(map(int, [x_min, y_min, x_max, y_max]))
            center_y = (y_min + y_max) / 2
            height = y_max - y_min
            all_detections.append({
                'bbox': bbox,
                'center_y': center_y,
                'height': height,
                'type': 'free'
            })

    if not all_detections:
        return []

    # Сортируем детекции по вертикальному центру
    all_detections.sort(key=lambda d: d['center_y'])
    return all_detections


def group_words_into_lines(
        horizontal_boxes: List[List[int]],
        free_boxes: List[List[List[int]]],
        y_threshold: float = 0.5,
        x_gap_threshold: float = 100,
        padding: int = 5
) -> list[Any] | list[list[int]]:
    """
    Группирует слова в строки и создает общие полигоны для каждой строки.
    
    Args:
        horizontal_boxes: Список горизонтальных боксов в формате [[x_min, x_max, y_min, y_max], ...]
        free_boxes: Список полигонов в формате [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
        y_threshold: Относительный порог для объединения по вертикали (доля средней высоты слов)
        x_gap_threshold: Максимальный разрыв по X для объединения слов в одну строку
        padding: Отступ для границ боксов строки
    
    Returns:
         Список горизонтальных боксов строк в формате [[x_min, y_min, x_max, y_max], ...]
    """
    all_detections = transform_bboxes(horizontal_boxes, free_boxes)

    # 2. Кластеризуем слова по строкам (по вертикали)
    # Вычисляем среднюю высоту для порога кластеризации
    avg_height = np.mean([d['height'] for d in all_detections])
    cluster_threshold = avg_height * y_threshold
    lines = []
    current_line = [all_detections[0]]
    
    for detection in all_detections[1:]:
        # Проверяем, принадлежит ли текущее слово к той же строке
        last_in_line = current_line[-1]
        vertical_distance = abs(detection['center_y'] - last_in_line['center_y'])
        
        if vertical_distance <= cluster_threshold:
            current_line.append(detection)
        else:
            # Начинаем новую строку
            lines.append(current_line)
            current_line = [detection]
    
    if current_line:
        lines.append(current_line)
    
    # 3. Для каждой строки создаем общий прямоугольник
    line_rectangles = []
    for line_words in lines:
        if not line_words:
            continue

        # Сортируем слова в строке по x_min
        line_words.sort(key=lambda d: d['bbox'][0])
        # Находим общие границы
        x_min = min([word['bbox'][0] for word in line_words])
        x_max = max([word['bbox'][2] for word in line_words])
        y_min = min([word['bbox'][1] for word in line_words])
        y_max = max([word['bbox'][3] for word in line_words])

        line_rectangles.append([x_min - padding, y_min - padding, 
                                   x_max + padding, y_max + padding])
    return line_rectangles


def extract_roi_images(
        image: np.ndarray,
        roi_polygons: List[List[int]],
        padding: int = 10
) -> List[Dict[str, Any]]:
    """
    Вырезает изображения строк по полигонам.
    
    Args:
        image: Исходное изображение
        roi_polygons: Список полигонов
        padding: Отступ от границ полигона при вырезании
        
    Returns:
        Список словарей с информацией о строках и вырезанных изображениях
    """
    results = []
    
    for i, polygon in enumerate(roi_polygons):
        x_min, y_min, x_max, y_max = map(int, polygon)
        # Добавляем небольшой отступ
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)

        if x_max <= x_min or y_max <= y_min:
            continue

        roi = image[y_min:y_max, x_min:x_max]
        results.append({
            'roi_number': i,
            'bounding_box': [x_min, y_min, x_max, y_max],
            'roi': roi
        })
    return results


def visualize_lines(
        image: np.ndarray,
        line_polygons: List[List[int]],
        output_path: str = None
) -> np.ndarray:
    """
    Визуализирует полигоны строк на изображении.
    
    Args:
        image: Исходное изображение
        line_polygons: Список полигонов строк
        output_path: Путь для сохранения результата
        
    Returns:
        Изображение с визуализацией
    """
    vis_image = image.copy()
    
    # Цвета для разных строк
    colors = [
        (255, 0, 0),    # Красный
        (0, 255, 0),    # Зеленый
        (0, 0, 255),    # Синий
        (255, 255, 0),  # Голубой
        (255, 0, 255),  # Пурпурный
        (0, 255, 255),  # Желтый
        (128, 0, 0),    # Темно-красный
        (0, 128, 0),    # Темно-зеленый
        (0, 0, 128),    # Темно-синий
    ]
    
    for i, polygon in enumerate(line_polygons):
        color = colors[i % len(colors)]
        x_min, y_min, x_max, y_max = map(int, polygon)

        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Рисуем полигон
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 3)
        # Подписываем номер строки
        cv2.putText(vis_image, f'Line {i}', (x_min, y_min), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def find_intervals_histogram(data: list[float], num_intervals: int, method: str = 'auto') -> list[dict[str, float]]:
    """
    Находит интервалы с помощью гистограммы

    Параметры:
    - data: массив чисел
    - num_intervals: количество интервалов для поиска
    - method: метод определения границ ('auto', 'sturges', 'scott', 'fd')

    Возвращает:
    - Список кортежей (начало, конец, количество точек)
    """
    if method == 'sturges':
        bins = int(np.ceil(np.log2(len(data))) + 1)
    elif method == 'scott':
        bin_width = 3.5 * np.std(data) / (len(data) ** (1 / 3))
        bins = int((np.max(data) - np.min(data)) / bin_width)
    elif method == 'fd':  # Freedman-Diaconis
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        bins = int((np.max(data) - np.min(data)) / bin_width) if bin_width > 0 else 1
    else:  # 'auto'
        bins = 'auto'

    hist, bin_edges = np.histogram(data, bins=bins)

    # Находим интервалы с наибольшим количеством точек
    sorted_indices = np.argsort(hist)[::-1][:num_intervals]
    intervals = []

    for idx in sorted_indices:
        intervals.append({
            'start': bin_edges[idx],
            'end': bin_edges[idx + 1],
            'count': hist[idx],
            'density': hist[idx] / (bin_edges[idx + 1] - bin_edges[idx])
        })
    return sorted(intervals, key=lambda x: x['start'])


def find_intervals_sliding_window(data, max_interval_width, min_points=3):
    """
    Находит интервалы с помощью скользящего окна с ограниченной шириной

    Параметры:
    - data: массив чисел
    - max_interval_width: максимальная ширина интервала
    - min_points: минимальное количество точек в интервале

    Возвращает:
    - Список интервалов
    """
    data_sorted = np.sort(data)
    n = len(data_sorted)
    intervals = []

    # Используем два указателя (метод скользящего окна)
    left = 0
    for right in range(n):
        # Сужаем окно слева, если превышена максимальная ширина
        while data_sorted[right] - data_sorted[left] > max_interval_width:
            left += 1

        # Если в окне достаточно точек
        if right - left + 1 >= min_points:
            current_start = data_sorted[left]
            current_end = data_sorted[right]
            current_count = right - left + 1
            current_density = current_count / (current_end - current_start)

            # Проверяем, не дублирует ли этот интервал предыдущий
            if not intervals or (current_start > intervals[-1]['end']):
                intervals.append({
                    'start': current_start,
                    'end': current_end,
                    'count': current_count,
                    'density': current_density,
                    'width': current_end - current_start
                })

    return sorted(intervals, key=lambda x: x['density'], reverse=True)


def process_handwritten_book_page(image_path, threshold: float = 0.3):
    """
    Полный пайплайн для обработки страницы рукописной книги
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # 1. Увеличиваем контраст
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    # 2. Удаляем шум
    enhanced = cv2.medianBlur(enhanced, 3)
    
    # Детекция текста
    reader = easyocr.Reader(['en'])
    horizontal_result, free_result = reader.detect(enhanced)
    
    # Группируем в строки
    lines = group_words_into_lines(horizontal_result[0], free_result[0], y_threshold=threshold, x_gap_threshold=200)
    # Вырезаем строки
    line_images = extract_roi_images(image, lines)
    return line_images, lines


def save_rois(lines, save_path):
    for i, line in enumerate(lines):
        cv2.imwrite(os.path.join(save_path, f'line_{i:03d}_roi.jpg'), line['roi'])


def main_detect_save(image_path, save_path, visualize_save_path=None):
    image = cv2.imread(image_path)
    line_images, lines = process_handwritten_book_page(im_path)
    save_rois(line_images, save_path)
    if visualize_save_path:
        visualize_lines(image, lines, visualize_save_path)


if __name__ == "__main__":
    im_path = r"C:\Users\gudko\history_envs\easyocr_env\big_block.jpg"
    save_path = r"C:\Users\gudko\history_envs\easyocr_env\big_block"
    visual_save_path = r"C:\Users\gudko\history_envs\easyocr_env\lines_detected.jpg"
    main_detect_save(im_path, save_path, visual_save_path)