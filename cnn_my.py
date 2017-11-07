import h5py
from keras import utils

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from helpers import NeptuneCallback
from deepsense import neptune

ctx = neptune.Context()

data = h5py.File("/input/cifar10.h5", 'r')
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

ctx.job.tags.append('cnn')
ctx.job.tags.append('my')

# create neural network architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(x_train, y_train,
          epochs=200,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20)])
