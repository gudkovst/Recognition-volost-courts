import keras
from keras import layers, models
from model_utils import *


class RecognitionModel(keras.Sequential):
    
    def __init__(self):
        super().__init__()

    def construct(self, input_shape, output_shape):
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
            
        h = self.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        show_graph('accuracy')
        show_graph('loss')        

    def test(self, X, y):
        prs = self.predict(X)
        calc_metrics(y, prs)

    def check_save(self, dir_save, name: str):
        if not type(name) == str:
            raise TypeError("name is not string: model don't save")
        ext = '.keras'
        name = name.split('.')[0] + ext
        if len(name) <= len(ext):
            raise TypeError("name is not defined: model don't save")
        self.save(os.path.join(dir_save, name))


class TestedModel(RecognitionModel):

    def __init__(self, data, name='model'):
        super().__init__()
        self.name = name
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.construct(self.X_train.shape[1:], self.y_train.shape[1])

    def learn(self, epochs=50):
        super().learn(self.X_train, self.y_train, epochs)

    def test(self):
        super().test(self.X_test, self.y_test)

    def test_eval(self):
        prs = self.predict(self.X_test)
        print('metrics of ', self.name, ' model')
        calc_metrics(self.y_test, prs)
        return prs

    