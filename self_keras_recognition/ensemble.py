from ensemble_utils import RecognitionSample
from glob import glob
from os.path import join as ospj
from recognition_model import *


class TestedEnsemble:

    methods = [get_data_nearest, get_data_bilinear, get_data_bicubic, get_data_lanczos]

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

    def __init__(self, config: set[(int, str)], alphabet_file: str):
        alphabet = np.load(alphabet_file)
        UniqueLabelEncoder().classes_ = alphabet
        self.alphabet_len = alphabet.shape[0]
        self.models = list()
        for rec in config:
            assert type(rec) == tuple and len(rec) == 2
            method, model_path = rec
            assert type(method) == int and type(model_path) == str 
            model = keras.saving.load_model(model_path)
            assert model.output_shape[1] == self.alphabet_len
            self.models.append((model.input_shape[1:], method, model))

    def get_data_config(self):
        return [rec[:2] for rec in self.models]
        
    def predict(self, data): # data: RecognitionSample
        self.predicts = np.zeros(data.get_params(), dtype=float)
        for rec in self.models:
            key = rec[:2]
            X = data.get_key_data(key)
            model = rec[2]
            self.predicts += model.predict(X)
        return self.predicts

    def _decode_predict(self, predicts, whitespace_indexes):
        prs = inverse_label(predicts)
        for num, white_index in enumerate(whitespace_indexes):
            prs = np.insert(prs, white_index + num, ' ')
        return prs

    def recognize(self, path):
        data_config = self.get_data_config()
        sample = RecognitionSample(data_config, self.alphabet_len)
        sample.construct(path)
        predicts = self.predict(sample) #TODO: write to file
        return self._decode_predict(predicts, sample.whitespace_index)
        
