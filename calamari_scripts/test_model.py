import os
import matplotlib.pyplot as plt
from sklearn import metrics


def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter


def calc_labels(predict_path):
    trues = []
    preds = []
    for pred in os.listdir(predict_path):
        trues.append(get_liter(pred))
        with open(os.path.join(predict_path, pred), encoding='utf-8') as f:
            preds.append(f.read())
    return trues, preds


predict_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filtered_first\nearest\predict0"

def make_conf_matrix(trues, preds):
    confusion_matrix = metrics.confusion_matrix(trues, preds)
    liter_set = {get_liter(l) for l in os.listdir(predict_path)}
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=sorted(liter_set))
    cm_display.plot()
    plt.show()


trues, preds = calc_labels(predict_path)
make_conf_matrix(trues, preds)
print(metrics.f1_score(trues, preds, average='macro'))
