from keras import layers, models
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


def build_model():
    pass


def prepare_data(size: int):
    X, labels = load_distributed_images(ROOT, size)
    le = UniqueLabelEncoder()
    y = le.fit_transform(labels)
    X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X, y, random_state=313, test_size=0.15)
    X_nn_train = X_nn_train.reshape((X_nn_train.shape[0], size, size, 4))
    X_nn_test = X_nn_test.reshape((X_nn_test.shape[0], size, size, 4))
    y_nn_train = to_categorical(y_nn_train)
    y_nn_test = to_categorical(y_nn_test)
    return X_nn_train, X_nn_test, y_nn_train, y_nn_test


def save_model(model, dir_save, name: str):
    if not type(name) == str:
        raise TypeError("name is not string: model don't save")
    ext = '.keras'
    name = name.split('.')[0] + ext
    if len(name) <= len(ext):
        raise TypeError("name is not defined: model don't save")
    model.save(os.path.join(dir_save, name))


def test_model(model, X, y):
    prs = model.predict(X)
    preds = [np.argmax(p) for p in prs]
    y_test = [np.argmax(yi) for yi in y]
    le = UniqueLabelEncoder()
    labels = le.inverse_transform(y_test)
    predicts = le.inverse_transform(preds)
    print_metrics(labels, predicts)
    make_conf_matrix(labels, predicts, labels=np.unique(labels))

