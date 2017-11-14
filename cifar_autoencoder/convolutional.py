from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from helpers import NeptuneCallback, load_cifar10, model_summary
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('convolutional')
encoding_dim = ctx.params['encoding_dim']
epochs = ctx.params['epochs']
batch_size = ctx.params['batch_size']
optimizer = ctx.params['optimizer']

c_bottleneck = encoding_dim // 16

input_img = Input(shape=(32, 32, 3))
x = input_img
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(c_bottleneck, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
encoded = x
x = Conv2D(c_bottleneck, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
decoded = x

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
model_summary(autoencoder)

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# training
autoencoder.fit(x_train, x_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test, x_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test[:5])])
