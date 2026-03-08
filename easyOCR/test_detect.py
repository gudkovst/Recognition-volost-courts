import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])

# 1. Загрузим изображение
image_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\strings\train\133601-179.jpg"
#"C:\Users\gudko\history_envs\easyocr_env\frag.jpg"

image = cv2.imread(image_path)

# 2. Вызовем ТОЛЬКО детектор
# Метод `detect` возвращает два списка: 
# - horizontal_list: список горизонтальных боксов в формате [x_min, x_max, y_min, y_max]
# - free_list: список свободных полигонов (многоугольников) для повернутого текста
horizontal_boxes, free_boxes = reader.detect(image)

# 3. Для книг обычно подходит горизонтальный список
# horizontal_boxes[0] содержит сами боксы, остальное — служебная информация
if horizontal_boxes[0]:
    for bbox in horizontal_boxes[0]:
        x_min, x_max, y_min, y_max = bbox
        print(f"Горизонтальный бокс: x:[{x_min}-{x_max}], y:[{y_min}-{y_max}]")
        
        # Визуализация (опционально)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

print("_______________________________________________________________________________")
if free_boxes[0]:
    for bbox in free_boxes[0]:
        print(f"Свободный бокс: {bbox}]")
        
        # Визуализация (опционально)
        int_bbox = [(int(p[0]), int(p[1])) for p in bbox]
        cv2.polylines(image, [np.array(int_bbox)], True, (0, 0, 255), 5)

# 4. Сохраним результат с разметкой
cv2.imwrite('detected_block.jpg', image)
print(f"Найдено строк: {len(horizontal_boxes[0])}")
