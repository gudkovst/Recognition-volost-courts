import paddleOCR_config as config
from PaddleOCR import paddleocr
import matplotlib.pyplot as plt
import os

def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter


liter_picts = os.listdir(config.img_path)
liter_set = {get_liter(l) for l in liter_picts}

model_dir = os.path.join(config.base_dir, 'model')
ocr = paddleocr.PaddleOCR(rec_model_dir=model_dir, use_angle_cls=True, lang='ru') # need to run only once to download and load model into memory
img_path = os.path.join(config.img_path, 'а-1007s.png')
img = plt.imread(img_path)
print("________________")
print(img.shape)
result = ocr.ocr(img, cls=True)
print(result)
print(len(result))
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


'''# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
print(txts)'''
