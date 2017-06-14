# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import sys
sys.path.insert(0, '/home/yoyo/Desktop/keras-adversarial-master')
#import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
#mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, merge
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2
from keras.utils.io_utils import HDF5Matrix
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import BatchNormalization
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import LeakyReLU, Activation, Dropout, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import h5py
import os

import matplotlib.pyplot as plt


def model_generator(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1(1e-7)):
    nch = 256
    g_input = Input(shape=[latent_dim])
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
#    g_input = Input(shape=[latent_dim])
#    H = Dense(128 * 8 * 8)(g_input)
#    H = BatchNormalization()(H)
#    H = Activation('relu')(H)
#    H = Reshape((8,8,128))(H)
#    
##    H = UpSampling2D(size=(2,2))(H)
##    H = Conv2D(128, (3, 3), padding='same', activation='relu')(H)
##    H = BatchNormalization()(H)
##    H = Activation('relu')(H)
#    
#    H = UpSampling2D(size=(2,2))(H)
#    H = Conv2D(64, (3, 3), padding='same', activation='relu')(H)
#    H = BatchNormalization()(H)
#    H = Activation('relu')(H)
#    
#    H = UpSampling2D(size=(2,2))(H)
#    H = Conv2D(32, (3, 3), padding='same', activation='relu')(H)
#    H = BatchNormalization()(H)
#    H = Activation('relu')(H)
#    
#    H = UpSampling2D(size=(2,2))(H)
#    H = Conv2D(32, (3, 3), padding='same', activation='relu')(H)
#    H = BatchNormalization()(H)
#    H = Activation('relu')(H)
#    
#    H = Conv2D(3, (1, 1), padding='same', activation='relu')(H)
#    g_V = Activation('tanh')(H)
#    return Model(g_input, g_V)
#    return Sequential([
#        Dense(hidden_dim, name="generator_h1", input_dim=latent_dim, kernel_regularizer=reg()),
#        LeakyReLU(0.2),
#        Dense(hidden_dim, name="generator_h2", kernel_regularizer=reg()),
#        LeakyReLU(0.2),
#        Dense(np.prod(input_shape), name="generator_x_flat", kernel_regularizer=reg()),
#        Activation('sigmoid'),
#        Reshape(input_shape, name="generator_x")],
#        name="generator")


def model_encoder(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1(1e-7)):
    dropout_rate = 0.5
    d_input = Input(input_shape, name="input_x")
    nch = 128
    H = Conv2D(int(nch / 2), (3, 3), padding='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(nch, (3, 3), strides=(2, 2), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(2*nch, (3, 3), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(latent_dim)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(latent_dim, activation='sigmoid')(H)
    return Model(d_input, d_V)
#    x = Input(input_shape, name="x")
#    h = BatchNormalization(input_shape=input_shape)(x)
#    
#    h = Conv2D(32, (3, 3),strides=(2,2), padding='same', activation='relu')(h)
##    h = MaxPooling2D(pool_size=2)(h)
#    h = Dropout(0.25)(h)
#
#    h = Conv2D(64, (3, 3),strides=(2,2), padding='same', activation='relu')(h)
##    h = MaxPooling2D(pool_size=2)(h)
#    h = Dropout(0.25)(h)
#
#    h = Conv2D(128, (3, 3),strides=(2,2), padding='same', activation='relu')(h)
##    h = MaxPooling2D(pool_size=2)(h)
#    h = Dropout(0.25)(h)
#
##    h = Conv2D(256, (3, 3),strides=(2,2), padding='same', activation='relu')(h)
###    h = MaxPooling2D(pool_size=2)(h)
##    h = Dropout(0.25)(h)
#        
#    h = Flatten()(h)
#    mu = Dense(latent_dim, name="encoder_mu", kernel_regularizer=reg())(h)
#    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", kernel_regularizer=reg())(h)
#    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
#              output_shape=lambda p: p[0])
#    return Model(x, z, name="encoder")
#    
#    x = Input(input_shape, name="x")
#    h = Flatten()(x)
#    h = Dense(hidden_dim, name="encoder_h1", kernel_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    h = Dense(hidden_dim, name="encoder_h2", kernel_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    mu = Dense(latent_dim, name="encoder_mu", kernel_regularizer=reg())(h)
#    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", kernel_regularizer=reg())(h)
#    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
#              output_shape=lambda p: p[0])
#    return Model(x, z, name="encoder")


def model_discriminator(latent_dim, output_dim=1, hidden_dim=512,
                        reg=lambda: l1_l2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = BatchNormalization(mode=2)(z)
    h = Dense(hidden_dim, name="discriminator_h1", kernel_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(mode=2)(h)
    h = Dense(hidden_dim, name="discriminator_h2", kernel_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", kernel_regularizer=reg())(h)
    return Model(z, y)



path = "output/aae"
adversarial_optimizer = AdversarialOptimizerSimultaneous()
# z \in R^100
latent_dim = 100
img_channels = 3
validation_split_size = 0.2
# x \in R^{28x28}
img_resize = (28, 28)
batch_size = 64
input_shape = (img_resize[0], img_resize[1], img_channels)

# generator (z -> x)
generator = model_generator(latent_dim, input_shape)
# encoder (x ->z)
encoder = model_encoder(latent_dim, input_shape)
# autoencoder (x -> x')
autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
# discriminator (z -> y)
discriminator = model_discriminator(latent_dim)

# assemple AAE
x = encoder.inputs[0]
z = encoder(x)
xpred = generator(z)
zreal = normal_latent_sampling((latent_dim,))(x)
yreal = discriminator(zreal)
yfake = discriminator(z)
aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

# print summary of models
generator.summary()
encoder.summary()
discriminator.summary()
autoencoder.summary()

# build adversarial model
generative_params = generator.trainable_weights + encoder.trainable_weights
model = AdversarialModel(base_model=aae,
                         player_params=[generative_params, discriminator.trainable_weights],
                         player_names=["generator", "discriminator"])
model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                          player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                "xpred": "mean_squared_error"},
                          compile_kwargs={"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}})

# load planet data
h5_train_file = "results/train_jpg_rgb.h5"
h5_test_file = "results/test_jpg_rgb.h5"

with h5py.File(h5_train_file, "r") as f:
    N_train = f["x_train"].shape[0]
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}

N_split = int(round(N_train * (1-validation_split_size)))


x_train = HDF5Matrix(h5_train_file, "x_train", start=0, end=N_split)
x_train = np.array(x_train)
x_train = x_train[:,4:60:2,4:60:2,:]
#y_train = HDF5Matrix(h5_train_file, "y_train", start=0, end=N_split)

x_valid = HDF5Matrix(h5_train_file, "x_train", start=N_split, end=N_train)
x_valid = np.array(x_valid)
x_valid = x_valid[:,4:60:2,4:60:2,:]
#y_valid = HDF5Matrix(h5_train_file, "y_train", start=N_split, end=N_train)


# callback for image grid of generated samples
def generator_sampler():
    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    return generator.predict(zsamples).reshape((10, 10, img_resize[0], img_resize[1], img_channels))

generator_cb = ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)

# callback for image grid of autoencoded samples
def autoencoder_sampler():
    xsamples = n_choice(x_valid, 10)
    xrep = np.repeat(xsamples, 9, axis=0)
    xgen = autoencoder.predict(xrep).reshape((10, 9, img_resize[0], img_resize[1], img_channels))
    xsamples = xsamples.reshape((10, 1, img_resize[0], img_resize[1], img_channels))
    samples = np.concatenate((xsamples, xgen), axis=1)
    return samples

autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler)

# train network
# generator, discriminator; pred, yfake, yreal
n = x_train.shape[0]
y = [x_train, np.ones((n, 1)), np.zeros((n, 1)), x_train, np.zeros((n, 1)), np.ones((n, 1))]
ntest = x_valid.shape[0]
ytest = [x_valid, np.ones((ntest, 1)), np.zeros((ntest, 1)), x_valid, np.zeros((ntest, 1)), np.ones((ntest, 1))]
history = model.fit(x=x_train, y=y, 
                    validation_data=(x_valid, ytest), 
                    callbacks=[generator_cb, autoencoder_cb],
                    epochs=100, 
                    batch_size=batch_size, 
                    shuffle="batch",
                    verbose=2)

# save history
df = pd.DataFrame(history.history)
df.to_csv(os.path.join(path, "history.csv"))

# save model
encoder.save(os.path.join(path, "encoder.h5"))
generator.save(os.path.join(path, "generator.h5"))
discriminator.save(os.path.join(path, "discriminator.h5"))


##    encoder2 = load_model(os.path.join(path, "encoder.h5"))
##    generator2 = load_model(os.path.join(path, "generator.h5"))
##    discriminator2 = load_model(os.path.join(path, "discriminator.h5"))
#
for num in range(3):
    plt.figure(num)
    plt.subplot(221)
    plt.imshow(x_train[num].squeeze(),cmap="gray")
    vec_fake = encoder.predict(x_train[num:num+1])
    im_fake = generator.predict(vec_fake);
    plt.subplot(222)
    plt.imshow(im_fake.squeeze(),cmap="gray")
    plt.subplot(223)
    plt.plot(vec_fake.squeeze())
