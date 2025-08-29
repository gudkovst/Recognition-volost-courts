import keras
from keras import layers, models
from model_utils import *
import os


@keras.saving.register_keras_serializable()
class RecognitionModel(keras.Sequential):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct(self, input_shape: tuple[int], output_shape: int):
        self.add(layers.Input(shape=input_shape))
        self.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dense(output_shape, activation='softmax'))
        self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'f1_score'])

    def learn(self, X, y, epochs):
        def show_graph(t: str):
            data = h.history[t]
            epochs = range(1, len(data) + 1)
            plt.plot(epochs, data, 'b')
            plt.title('Training ' + t)
            plt.show()
            
        h = super().fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        show_graph('accuracy')
        show_graph('loss')    

    def test(self, X, y):
        prs = super().predict(X)
        calc_metrics(y, prs)

    def check_save(self, dir_save, name: str):
        if not type(name) == str:
            raise TypeError("name is not string: model don't save")
        ext = '.keras'
        name = name.split('.')[0] + ext
        if len(name) <= len(ext):
            raise TypeError("name is not defined: model don't save")
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)
        super().save(os.path.join(dir_save, name))

    def get_config(self):
        return super().get_config()


class TestedModel():

    def __init__(self, data, name: str = 'test_model', model: str = None):
        self.name = name
        self.X_train, self.X_test, self.y_train, self.y_test = data
        if not model:
            self.model = RecognitionModel()
            self.model.construct(self.X_train.shape[1:], self.y_train.shape[1])
        else:
            self.model = keras.saving.load_model(model)
        

    def learn(self, epochs=50):
        self.model.learn(self.X_train, self.y_train, epochs)

    def test(self):
        self.model.test(self.X_test, self.y_test)

    def test_eval(self):
        prs = self.model.predict(self.X_test)
        print('metrics of ', self.name, ' model')
        calc_metrics(self.y_test, prs)
        return prs

    def save(self, dir_name):
        self.model.check_save(dir_name, self.name)
