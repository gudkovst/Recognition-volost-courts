from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import *


class UniqueLabelEncoder(LabelEncoder):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UniqueLabelEncoder, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass


ROOT = r'C:\Users\gudko\history_envs\data\ALL_LITERS\__почерк1'


def prepare_data(size: int, interpolator=Image.BICUBIC, test_part=0.15):
    X, labels = load_distributed_images(ROOT, size, interpolator)
    le = UniqueLabelEncoder()
    y = le.fit_transform(labels)
    X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X, y, random_state=313, test_size=test_part)
    X_nn_train = X_nn_train.reshape((X_nn_train.shape[0], size, size, 4))
    X_nn_test = X_nn_test.reshape((X_nn_test.shape[0], size, size, 4))
    y_nn_train = to_categorical(y_nn_train)
    y_nn_test = to_categorical(y_nn_test)
    return X_nn_train, X_nn_test, y_nn_train, y_nn_test


def get_data_nearest(size: int, test_part=0.15):
    return prepare_data(size, Image.NEAREST, test_part)


def get_data_bilinear(size: int, test_part=0.15):
    return prepare_data(size, Image.BILINEAR, test_part)


def get_data_bicubic(size: int, test_part=0.15):
    return prepare_data(size, Image.BICUBIC, test_part)


def get_data_lanczos(size: int, test_part=0.15):
    return prepare_data(size, Image.LANCZOS, test_part)


def inverse_label(labels):
    le = UniqueLabelEncoder()
    return le.inverse_transform([np.argmax(l) for l in labels])


def print_metrics(trues, preds):
    f1 = metrics.f1_score(trues, preds, average='macro')
    precision = metrics.precision_score(trues, preds, average='macro')
    recall = metrics.recall_score(trues, preds, average='macro')
    print(f"precision: {precision}\nrecall: {recall}\nF1: {f1}")


def make_conf_matrix(trues, preds, labels):
    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=sorted(labels))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=sorted(labels))
    cm_display.plot()
    plt.show()


def calc_metrics(y, prs):
    labels = inverse_label(y)
    predicts = inverse_label(prs)
    print_metrics(labels, predicts)
    make_conf_matrix(labels, predicts, labels=np.unique(labels))
