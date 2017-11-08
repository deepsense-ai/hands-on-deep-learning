from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from helpers import NeptuneCallback, load_cifar10
from deepsense import neptune

ctx = neptune.Context()
ctx.job.tags.append('logistic-regression')

# create neural network architecture
model = Sequential()

model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# training
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20)])
