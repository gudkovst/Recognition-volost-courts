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
            
        h = self.fit(X, y, epochs=epochs, batch_size=32)
        show_graph('accuracy')
        show_graph('loss')        

    def test(self, X, y):
        prs = self.predict(X)
        labels = inverse_label(y)
        predicts = inverse_label(prs)
        print_metrics(labels, predicts)
        make_conf_matrix(labels, predicts, labels=np.unique(labels))

    def check_save(self, dir_save, name: str):
        if not type(name) == str:
            raise TypeError("name is not string: model don't save")
        ext = '.keras'
        name = name.split('.')[0] + ext
        if len(name) <= len(ext):
            raise TypeError("name is not defined: model don't save")
        self.save(os.path.join(dir_save, name))
        