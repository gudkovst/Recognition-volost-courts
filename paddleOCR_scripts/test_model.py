import paddleOCR_config as config
from PaddleOCR import paddleocr
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import subprocess


def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter


liter_picts = os.listdir(config.test_img_path)
liter_set = {get_liter(l) for l in liter_picts}


def model_recognition(imgs_path: list):
    cmd = f'{config.python_exe_path} {config.rec_infer_path} -c {config.model_config} -o Global.pretrained_model={config.model_path}'
    for img_name in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, img_name)
        infer_img = f" Global.infer_img={img_path}"
        cmd_img = cmd + infer_img
        subprocess.run(cmd_img, stdout=subprocess.DEVNULL)

#model_recognition(config.test_img_path)

def calc_labels(predict_filename) -> [list, list]:
    preds = []
    trues = []
    with open(predict_filename, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
        for line in lines:
            record = line.split('\t')
            name_liter = os.path.basename(record[0])
            trues.append(get_liter(name_liter))
            preds.append(record[1])
    return trues, preds


predict_file = r"C:\Users\User\jupyter\paddleOCR_train_env\output\rec\predicts_rec.txt"

def make_conf_matrix():
    trues, preds = calc_labels(predict_file)
    confusion_matrix = metrics.confusion_matrix(trues, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=sorted(liter_set))
    cm_display.plot()
    plt.show()


#make_conf_matrix()
trues, preds = calc_labels(predict_file)
print(metrics.f1_score(trues, preds, average='macro'))
