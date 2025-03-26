import os
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import get_liter


def calc_labels(predict_path):
    trues = []
    preds = []
    for pred in os.listdir(predict_path):
        trues.append(get_liter(pred))
        with open(os.path.join(predict_path, pred), encoding='utf-8') as f:
            preds.append(f.read())
    return trues, preds


predict_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filter_24_03\repeat\predict0\lanczos"

def print_metrics(trues, preds):
    f1 = metrics.f1_score(trues, preds, average='macro')
    precision = metrics.precision_score(trues, preds, average='macro')
    recall = metrics.recall_score(trues, preds, average='macro')
    print(f"precision: {precision}\nrecall: {recall}\nF1: {f1}")


def make_conf_matrix(trues, preds):
    confusion_matrix = metrics.confusion_matrix(trues, preds)
    liter_set = {get_liter(l) for l in os.listdir(predict_path)}
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=sorted(liter_set))
    cm_display.plot()
    plt.show()


trues, preds = calc_labels(predict_path)
print_metrics(trues, preds)
make_conf_matrix(trues, preds)
