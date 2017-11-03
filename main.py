import h5py
import numpy as np
from keras.utils import np_utils
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.callbacks import Callback

from deepsense import neptune


def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) == 2
    if array.max() <= 1 and autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)


class NeptuneCallback(Callback):
    def __init__(self, images_per_epoch=-1):
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        ctx.job.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        ctx.job.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.job.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        ctx.job.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])

        # Predict the digits for images of the test set.
        validation_predictions = model.predict_classes(X_test)
        scores = model.predict(X_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        image_per_epoch = 0
        for index, (prediction, actual) in enumerate(zip(validation_predictions, Y_test.argmax(axis=1))):
            if prediction != actual:
                if image_per_epoch == self.images_per_epoch:
                    break
                image_per_epoch += 1

                ctx.job.channel_send('false_predictions', neptune.Image(
                    name='[{}] pred: {} true: {}'.format(self.epoch_id, categories[prediction], categories[actual]),
                    description="\n".join([
                        "{} {:5.1f}% {}".format(letters[i], 100 * score, "!!!" if i == actual else "")
                        for i, score in enumerate(scores[index])]),
                    data=array_2d_to_image(X_test[index,:,:,0])))


data = h5py.File("../input/cifar10.h5", 'r')
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

resolution = 32
classes = 10
categories = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

ctx = neptune.Context()

# create neural network architecture
model = Sequential()

model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(x_train, x_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, x_test),
          verbose=2,
          callbacks=[NeptuneCallback(images_per_epoch=20)])
