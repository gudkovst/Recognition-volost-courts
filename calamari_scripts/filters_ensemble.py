import os
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import get_liter


def vote4(preds: dict) -> str:
    assert len(preds) == 4
    d = dict()
    for p in preds:
        label = preds[p]
        d[label] = d.get(label, 0) + 1
    skeys = sorted(d, key=lambda x: d[x], reverse=True)
    votes = d[skeys[0]]
    count = 0
    for v in d:
        if d[v] == votes:
            count += 1
    if count == 1:
        return skeys[0]
    return preds['lanczos']


def calc_labels(predict_paths):
    def get_filter(path: str) -> str:
        return os.path.basename(path)
    
    trues = []
    preds = []
    for pred_file in os.listdir(predict_paths[0]):
        trues.append(get_liter(pred_file))
        local_preds = dict()
        for pred_path in predict_paths:
            pred_filename = os.path.join(pred_path, pred_file)
            filt = get_filter(pred_path)
            with open(pred_filename, encoding="utf-8") as f:
                local_preds[filt] = f.read()
        res_vote = vote4(local_preds)
        if len(res_vote) == 1:
            preds.append(res_vote[0])
    return trues, preds


def print_metrics(trues, preds):
    f1 = metrics.f1_score(trues, preds, average='macro')
    precision = metrics.precision_score(trues, preds, average='macro')
    recall = metrics.recall_score(trues, preds, average='macro')
    print(f"precision: {precision}\nrecall: {recall}\nF1: {f1}")
    

def make_conf_matrix(trues, preds):
    confusion_matrix = metrics.confusion_matrix(trues, preds)
    #liter_set = {get_liter(l) for l in os.listdir(os.path.join(data_path, size, 'nearest', 'predict0'))}
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.show()


data_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filter_24_03\not_repeat"
filters = ['nearest', 'bilinear', 'bicubic', 'lanczos']

size = '200x200'

predict_paths = [os.path.join(data_path, 'predict0', filt) for filt in filters]
trues, preds = calc_labels(predict_paths)
print_metrics(trues, preds)
make_conf_matrix(trues, preds)
