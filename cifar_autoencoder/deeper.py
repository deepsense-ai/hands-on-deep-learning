from keras.models import Model
from keras.layers import Input, Dense
from helpers import NeptuneCallback, load_cifar10, model_summary
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('deeper')
encoding_dim = ctx.params['encoding_dim']

# 32 * 32 * 3 == 3072
# this is our input placeholder
input_img = Input(shape=(32 * 32 * 3,))
x = input_img
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(encoding_dim, activation='relu')(x)
encoded = x
x = Dense(512, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
decoded = Dense(32 * 32 * 3, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
model_summary(autoencoder)

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar10()
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)

# training
autoencoder.fit(x_train, x_train,
          epochs=50,
          batch_size=256,
          validation_data=(x_test, x_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test[:5])])
