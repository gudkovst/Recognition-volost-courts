from glob import glob
from os.path import join as ospj
from recognition_model import *


methods = [get_data_nearest, get_data_bilinear, get_data_bicubic, get_data_lanczos]


class Ensemble:
    
    def __init__(self, size):
        self.size = size
        data = methods[0](size)
        _, _, _, self.y_test = data
        self.models = [TestedModel(data, 'nearest')]
        for method in methods[1:]:
            name = method.__name__.split('_')[-1]
            self.models.append(TestedModel(method(size), name))
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


class LoadedEnsemble(Ensemble):

    def __init__(self, path):
        for name in glob(ospj(path, '*.keras')):
            pass
    

class MetaEnsemble:

    def __init__(self, *sizes):
        _, _, _, self.y_test = methods[0](1)
        self.ensembles = []
        for size in sizes:
            self.ensembles.append(Ensemble(size))
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
