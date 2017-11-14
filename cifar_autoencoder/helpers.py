import h5py
from keras import utils
import numpy as np
from PIL import Image
from keras.callbacks import Callback
from deepsense import neptune

def load_cifar10(filepath="/public/cifar/cifar10.h5"):
    data = h5py.File(filepath, 'r')
    x_train = data['x_train'].value
    y_train = data['y_train'].value
    x_test = data['x_test'].value
    y_test = data['y_test'].value

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    print("CIFAR-10 data loaded from {}.".format(filepath))

    return (x_train, y_train), (x_test, y_test)

ctx = neptune.Context()

def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) in [2, 3]
    if array.max() <= 1 and autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)

def model_summary(model):
    print("Model created successfully.")
    print(model.summary())
    ctx.channel_send('n_layers', len(model.layers))
    ctx.channel_send('n_parameters', model.count_params())

class NeptuneCallback(Callback):
    def __init__(self, x_test):
        self.epoch_id = 0
        self.x_test = x_test

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # Logging numeric channels
        ctx.channel_send('Loss training', self.epoch_id, logs['loss'])
        ctx.channel_send('Loss validation', self.epoch_id, logs['val_loss'])

        # Showing autoencoded pictures
        autoencoded = self.model.predict(self.x_test)

        # for x, score in zip(autoencoded, scores):
        for i, x in enumerate(autoencoded):
            x0 = self.x_test[i:i+1]
            score = self.model.evaluate(x0, x0, batch_size=1, verbose=0)
            ctx.channel_send('autoencoded', neptune.Image(
                name="[{}] {:.2f}".format(self.epoch_id, score),
                description="Epoch: {}\nLoss: {:.2f}".format(self.epoch_id, score),
                data=array_2d_to_image(x.reshape(32, 32, 3))))
