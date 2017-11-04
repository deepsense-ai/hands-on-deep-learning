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

# create neural network architecture
model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
