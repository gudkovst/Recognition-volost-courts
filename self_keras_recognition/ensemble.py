from ensemble_utils import RecognitionSample
from glob import glob
from os.path import join as ospj
from recognition_model import *


methods = [get_data_nearest, get_data_bilinear, get_data_bicubic, get_data_lanczos]


class TestedEnsemble:
    
    def __init__(self, size):
        self.size = size
        self.models = []
        for method in methods:
            name = method.__name__.split('_')[-1]
            data = method(size)
            self.models.append(TestedModel(data, name))
        _, _, _, self.y_test = data
        self.predicts = None

    def learn(self, epochs=50):
        for model in self.models:
            model.learn(epochs)

    def predict(self):
        if self.predicts is not None:
            return self.predicts
        self.predicts = np.zeros(self.y_test.shape, dtype=float)
        for model in self.models:
            self.predicts += model.test_eval()
        return self.predicts

    def test(self):
        if self.predicts is None:
            self.predict()
        print('metrics of ensemble ', self.size)
        calc_metrics(self.y_test, self.predicts)

    def save(self):
        path = ospj('recognition_models', str(self.size))
        for model, method in zip(self.models, methods):
            model.check_save(path, method.__name__.split('_')[-1])


class MetaEnsemble:

    def __init__(self, *sizes):
        _, _, _, self.y_test = methods[0](1)
        self.ensembles = []
        for size in sizes:
            self.ensembles.append(TestedEnsemble(size))
        self.predicts = None

    def learn(self, epochs=50):
        for ens in self.ensembles:
            ens.learn(epochs)

    def predict(self):
        self.predicts = np.zeros(self.y_test.shape, dtype=float)
        for ens in self.ensembles:
            self.predicts += ens.predict()
        return self.predicts

    def test(self):
        if self.predicts is None:
            self.predict()
        print('metrics of metaensemble')
        calc_metrics(self.y_test, self.predicts)


class FittedMetaEnsemble(MetaEnsemble):
    
    def __init__(self, *ensembles):
        _, _, _, self.y_test = methods[0](1)
        self.ensembles = list(ensembles)
        self.predicts = None

    def learn(self, epochs):
        pass


class LoadedEnsemble:

    def __init__(self, config: dict[(int, int), str], alphabet_len):
        self.alphabet_len = alphabet_len
        self.models = {}
        for key in config:
            assert type(key) == tuple and len(key) == 2 and type(key[0]) == type(key[1]) == int
            model = keras.saving.load_model(config[key])
            self.models[key] = model

    def get_data_config(self):
        return self.models.keys()
        
    def predict(self, data): # data: RecognitionSample
        self.predicts = np.zeros(data.get_params(), dtype=float)
        print(self.predicts.shape)
        for key in self.models:
            X = data.get_key_data(key)
            print(X.shape)
            self.predicts += self.models[key].predict(X)
        return self.predicts

    def recognize(self, path):
        data_config = self.get_data_config()
        sample = RecognitionSample(data_config, self.alphabet_len)
        sample.construct(path)
        predicts = self.predict(sample) #TODO: write to file
        return inverse_label(predicts)