from keras.models import Model
from keras.layers import Input, Dense
from helpers import NeptuneCallback, load_cifar10, model_summary
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('shallow')

# this is the size of our encoded representations
encoding_dim = 256

# this is our input placeholder
input_img = Input(shape=(32 * 32 * 3,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(32 * 32 * 3, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

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
