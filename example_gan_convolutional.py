import sys
sys.path.insert(0, '/home/yoyo/Desktop/keras-adversarial-master')

import matplotlib as plt
import h5py
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation
from keras.layers import Dense, Conv2D, Reshape
from keras.models import Model
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist
import pandas as pd
import numpy as np
import keras.backend as K
from keras_adversarial.legacy import BatchNormalization
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.utils.io_utils import HDF5Matrix


def leaky_relu(x):
    return K.relu(x, 0.2)


def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14)(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Reshape((14, 14, nch))(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding='same')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Conv2D(int(nch / 4), (3, 3), padding='same')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Conv2D(3, (1, 1), padding='same')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)


def model_discriminator(input_shape=(28, 28, 3), dropout_rate=0.5):
    d_input = Input(input_shape, name="input_x")
    nch = 128
    H = Conv2D(int(nch / 2), (5, 5), strides=(2, 2), padding='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(nch, (5, 5), strides=(2, 2), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


def generator_sampler(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = generator.predict(zsamples)
        return gen.reshape((10, 10, 28, 28, 3))

    return fun


# z \in R^100
latent_dim = 100
# x \in R^{28x28}
input_shape = (28, 28, 3)

# generator (z -> x)
generator = model_generator()
# discriminator (x -> y)
discriminator = model_discriminator(input_shape=input_shape)
# gan (x - > yfake, yreal), z generated on GPU
gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

# print summary of models
generator.summary()
discriminator.summary()
gan.summary()

# build adversarial model
model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights, discriminator.trainable_weights],
                         player_names=["generator", "discriminator"])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss='binary_crossentropy')

# train model
generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",
                                 generator_sampler(latent_dim, generator))

h5_train_file = "results/train_jpg_rgb.h5"
h5_test_file = "results/test_jpg_rgb.h5"
validation_split_size = 0.2

with h5py.File(h5_train_file, "r") as f:
    N_train = f["x_train"].shape[0]
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}

N_split = int(round(N_train * (1-validation_split_size)))

xtrain = HDF5Matrix(h5_train_file, "x_train", start=0, end=N_split)
xtrain = np.array(xtrain)
xtrain = xtrain[:,4:60:2,4:60:2,:]

#y_train = HDF5Matrix(h5_train_file, "y_train", start=0, end=N_split)

xtest = HDF5Matrix(h5_train_file, "x_train", start=N_split, end=N_train)
xtest = np.array(xtest)
xtest = xtest[:,4:60:2,4:60:2,:]

#    xtrain, xtest = mnist_data()
#    xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
#    xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))

y = gan_targets(xtrain.shape[0])
ytest = gan_targets(xtest.shape[0])

history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb], epochs=100,
                    shuffle="batch", batch_size=64, verbose=2)

df = pd.DataFrame(history.history)
df.to_csv("output/gan_convolutional/history.csv")

generator.save("output/gan_convolutional/generator.h5")
discriminator.save("output/gan_convolutional/discriminator.h5")
