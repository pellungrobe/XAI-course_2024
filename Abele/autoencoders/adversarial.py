import numpy as np
from datetime import datetime

import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, GaussianNoise, Add, AveragePooling2D, Dropout
from keras.layers import LeakyReLU, ReLU
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.datasets import mnist
import keras
from abc import abstractmethod
import tensorflow_addons as tfa

from autoencoders.autoencoder import Autoencoder
from autoencoders.discrimination import MinibatchDiscrimination


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon

def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

def l1diff(x,y):
    return np.mean(np.abs(x - y))

class AdversarialAutoencoder(Autoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=512, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoder, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                     store_intermediate, save_graph, path, name)

    @abstractmethod
    def build_encoder(self):
        return

    @abstractmethod
    def build_decoder(self):
        return

    @abstractmethod
    def build_discriminator(self):
        return

    def init(self):

        optimizer_d = Adam(0.0002, 0.5)
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        #x = Input(shape=(64,64,3))
        x = Input(shape=self.shape)
        # The generator takes the image, encodes it and reconstructs it from the encoding
        lx = self.encoder(x)   # latent representation (latent x)
        tx = self.decoder(lx)  # reconstructed record (tilde x)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(lx)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.autoencoder = Model(x, [tx, validity])

        self.autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

        if self.verbose:
            self.autoencoder.summary()

        #if self.save_graph:
            #plot_model(self.encoder, to_file='%s%s_encoder.png' % (self.path, self.name))
            #plot_model(self.decoder, to_file='%s%s_decoder.png' % (self.path, self.name))
            #plot_model(self.autoencoder, to_file='%s%s_adversarial.png' % (self.path, self.name))
            #plot_model(self.discriminator, to_file='%s%s_discriminator.png' % (self.path, self.name))

    def fit(self, X, Y, epochs=30000, batch_size=128, sample_interval=100):

        self.init()
        X = self.img_normalize(X) #afhq

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        half_batch = int(batch_size / 2)
        past = datetime.now()
        test_array = []
        np.save('/content/gdrive/My Drive/Colab Notebooks/ABELE/ABELE_master/aemodels/aae/model.npy', np.array(test_array))
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X.shape[0], batch_size) #afhq

            Xs = X[idx] #afhq
            #Xs = X

            latent_fake = self.encoder.predict(Xs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            Z = np.concatenate([latent_fake, latent_real])
            y = np.concatenate([fake, valid])
            d_loss = self.discriminator.train_on_batch(Z, y)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.autoencoder.train_on_batch(Xs, [Xs, valid])


            now = datetime.now()
            # Plot the progress

            if self.verbose and epoch % sample_interval == 0:
                #print("Epoch %d/%d, %.2f [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                    #epoch, epochs, (now - past).total_seconds(), d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

                #X_test = next(Y)
                X_test = Y
                X_test_ae = self.decode(self.encode(X_test))
                rmse_ = rmse(X_test, X_test_ae)
                l1diff_ = l1diff(X_test, X_test_ae)

                test_prova = np.load('/content/gdrive/My Drive/Colab Notebooks/ABELE/ABELE_master/aemodels/aae/model.npy')
                test_prova = np.append(test_prova, [d_loss[0], d_loss[1], g_loss[0], g_loss[1], rmse_, l1diff_])
                np.save('/content/gdrive/My Drive/Colab Notebooks/ABELE/ABELE_master/aemodels/aae/model.npy', np.array(test_prova))
                print("Epoch %d/%d, %.2f [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f] [l2 test: %f, l1 test: %f]" % (
                    epoch, epochs, (now - past).total_seconds(), d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], rmse_, l1diff_))
            past = now

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and self.store_intermediate:
                self.sample_images(epoch)
                self.save_model()

        if self.store_intermediate:
            self.sample_images(epochs)

        self.save_model()


class AdversarialAutoencoderMnist(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=512, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderMnist, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                          store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return (X.astype(np.float32) - 127.5) / 127.5

    def img_denormalize(self, X):
        return (X * 127.5 + 127.5).astype(int)

    def build_encoder(self):

        x = Input(shape=self.shape)
        h = Flatten()(x)
        h = Dense(self.hidden_dim)(h)
        #h = LeakyReLU(alpha=self.alpha)(h)
        h = ReLU()(h)
        h = Dense(self.hidden_dim)(h)
        #h = LeakyReLU(alpha=self.alpha)(h)
        h = ReLU()(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(sampling)([mu, log_var])

        model = Model(x, latent_repr)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(Dense(self.hidden_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(Dense(self.input_dim, activation='tanh'))
        model.add(Reshape(self.shape))
        if self.verbose:
            model.summary()

        z = Input(shape=(self.latent_dim,))
        tx = model(z)

        return Model(z, tx)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(Dense(self.hidden_dim//2))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

class AdversarialAutoencoderMnist_war(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=512, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderMnist_war, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                          store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return (X.astype(np.float32) - 127.5) / 127.5

    def img_denormalize(self, X):
        return (X * 127.5 + 127.5).astype(int)

    def build_encoder(self):

        x = Input(shape=self.shape)
        h = Flatten()(x)
        h = X_dense(self.hidden_dim)(h)
        h = Bias()(h)
        #h = LeakyReLU(alpha=self.alpha)(h)
        #h = ReLU()(h)
        h = Dense(self.hidden_dim)(h)
        h = Bias()(h)
        #h = LeakyReLU(alpha=self.alpha)(h)
        #h = ReLU()(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(sampling)([mu, log_var])

        model = Model(x, latent_repr)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(X_dense(self.hidden_dim))
        model.add(Bias())
        #model.add(LeakyReLU(alpha=self.alpha))
        #model.add(ReLU())
        model.add(Dense(self.input_dim, activation='tanh'))
        model.add(Reshape(self.shape))
        if self.verbose:
            model.summary()

        z = Input(shape=(self.latent_dim,))
        tx = model(z)

        return Model(z, tx)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(ReLU())
        model.add(X_dense(self.hidden_dim))
        model.add(Bias())
        #model.add(LeakyReLU(alpha=self.alpha))
        #model.add(ReLU())
        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

class AdversarialAutoencoderCifar10(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=128, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderCifar10, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                            store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return X.astype(np.float32) / 255.0

    def img_denormalize(self, X):
        return (X * 255).astype(int)

    def build_encoder(self):

        x = Input(shape=self.shape)

        # h = Conv2D(64, (3, 3), padding='same')(x)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(3, kernel_size=(2, 2), padding='same', activation='relu')(x)
        h = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(h)
        h = Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(h)
        h = Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(h)

        h = Flatten()(h)
        h = Dense(self.hidden_dim, activation='relu')(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        lx = Lambda(sampling)([mu, log_var])

        model = Model(x, lx)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        lx = Input(shape=(self.latent_dim,))

        h = Dense(self.hidden_dim, activation='relu')(lx)
        # h = Dense(self.hidden_dim * 2, activation='relu')(h)
        # h = Reshape((4, 4, 16))(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(64, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(3, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('sigmoid')(h)

        h = Dense(32 * 16 * 16, activation='relu')(h)
        h = Reshape((16, 16, 32))(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(h)
        h = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(h)
        h = Conv2D(3, kernel_size=2, padding='valid', activation='sigmoid')(h)

        model = Model(lx, h)
        if self.verbose:
            model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        model.add(ReLU())
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(self.hidden_dim//2))
        model.add(ReLU())
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

class AdversarialAutoencoderCifar10_war(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=128, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderCifar10_war, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                            store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return X.astype(np.float32) / 255.0

    def img_denormalize(self, X):
        return (X * 255).astype(int)

    def build_encoder(self):

        x = Input(shape=self.shape)

        # h = Conv2D(64, (3, 3), padding='same')(x)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)

        h = M_Conv2D(3, kernel_size=(2, 2), padding='SAME')(x)
        h = M_Conv2D(32, kernel_size=(2, 2), padding='SAME', strides=(2, 2))(h)
        h = M_Conv2D(32, kernel_size=(3, 3), padding='SAME', strides=1)(h)
        h = M_Conv2D(32, kernel_size=(3, 3), padding='SAME', strides=1)(h)

        h = Flatten()(h)
        h = X_dense(self.hidden_dim)(h)
        h = Bias()(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        lx = Lambda(sampling)([mu, log_var])

        model = Model(x, lx)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        lx = Input(shape=(self.latent_dim,))

        h = X_dense(self.hidden_dim)(lx)
        h = Bias()(h)
        # h = Dense(self.hidden_dim * 2, activation='relu')(h)
        # h = Reshape((4, 4, 16))(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(64, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(3, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('sigmoid')(h)

        h = X_dense(32 * 16 * 16)(h)
        h = Bias()(h)
        h = Reshape((16, 16, 32))(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(h)
        h = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(h)
        h = Conv2D(3, kernel_size=2, padding='valid', activation='sigmoid')(h)

        model = Model(lx, h)
        if self.verbose:
            model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(keras.Input(shape=self.latent_dim))
        model.add(X_dense(self.hidden_dim))
        model.add(Bias())
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(X_dense(self.hidden_dim//2))
        model.add(Bias())
        #model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

class AdversarialAutoencoderProstate(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=8, hidden_dim=128, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderProstate, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                            store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return (X.astype(np.float32) - 127.5) / 127.5

    def img_denormalize(self, X):
        return (X * 127.5 + 127.5).astype(int)

    def build_encoder(self):

        bm = 0.95
        kernel_reg_coefficient = 0.5e-4
        #kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None

        x = Input(shape=self.shape)
        #h = GaussianNoise(stddev=0.05)(x, training=True)
        # h = Conv2D(64, (3, 3), padding='same')(x)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)


        h = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(x)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(h)

        h = Conv2D(32, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2D(256, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2D(512, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        #h = MaxPooling2D((2, 2), padding='same')(h)
        h = GlobalAveragePooling2D()(h)
        #h = Flatten()(h)
        #h = Dense(self.hidden_dim/4, activation='relu')(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        lx = Lambda(sampling)([mu, log_var])

        model = Model(x, lx)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        bm = 0.95

        kernel_reg_coefficient = 0.5e-4
        #kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None
        lx = Input(shape=(self.latent_dim,))

        h = Dense(self.hidden_dim/4, activation='relu')(lx)
        # h = Dense(self.hidden_dim * 2, activation='relu')(h)
        # h = Reshape((4, 4, 16))(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(64, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(3, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('sigmoid')(h)

        h = Dense(self.hidden_dim/2, activation='relu', kernel_regularizer = kernel_regularizer)(h)
        h = Dense(self.hidden_dim, activation='relu', kernel_regularizer = kernel_regularizer)(h)
        h = Reshape((6, 6, 32))(h)

        h = Conv2DTranspose(512, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        # h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(512, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        # h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(256, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(256, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        #h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(128, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(128, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        #h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(64, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(64, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        #h = UpSampling2D((2, 2))(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(32, kernel_size=4, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        #h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        #h = UpSampling2D((2, 2))(h)

        #h = Conv2D(128, (3, 3), padding='same')(h)
        #h = BatchNormalization(momentum=bm)(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        #h = Activation('relu')(h)
        #h = Conv2D(128, (3, 3), padding='same')(h)
        #h = BatchNormalization(momentum=bm)(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        #h = Activation('relu')(h)
        #h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(1, kernel_size=3, padding='same', strides=1,
                            kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)
        #h = Conv2D(1, (3, 3), padding='same')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('tanh')(h)




        '''
        h = Conv2DTranspose(16, (3, 3), strides=(2,2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2D(1, (3, 3), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        '''

        model = Model(lx, h)
        if self.verbose:
            model.summary()

        return model

    def build_discriminator(self):

        bm = 0.95
        kernel_reg_coefficient = 2e-4
        #kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None
        model = Sequential()
        model.add(Dense(2048, input_dim=self.latent_dim))
        #model.add(BatchNormalization(momentum=bm))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(2048))
        #model.add(BatchNormalization(momentum=bm))
        model.add(LeakyReLU(alpha=self.alpha))


        #model.add(Dense(self.hidden_dim*2, input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=self.alpha))
        #model.add(Dense(self.hidden_dim))
        #model.add(LeakyReLU(alpha=self.alpha))

        '''

        model.add(Dense(self.hidden_dim , input_dim=self.latent_dim, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Reshape((24, 24, 2)))


        model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform"))
        #model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))
        # model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(256, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))
        # model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))

        model.add(GlobalAveragePooling2D())
        model.add(MinibatchDiscrimination(20, 5))
        
        '''
        '''
        model.add(Flatten())

        model.add(Dense(512, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(MinibatchDiscrimination(20, 5))

        model.add(Dense(512, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        '''


        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)


class AdversarialAutoencoderafhq(AdversarialAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=8, hidden_dim=128, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        super(AdversarialAutoencoderafhq, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                             store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return (X.astype(np.float32) - 127.5) / 127.5

    def img_denormalize(self, X):
        return (X * 127.5 + 127.5).astype(int)

    def build_encoder(self):

        bm = 0.95
        kernel_reg_coefficient = 0.5e-4
        # kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None

        x = Input(shape=self.shape)
        #x = Input(shape=(64,64,3))
        # h = GaussianNoise(stddev=0.05)(x, training=True)
        # h = Conv2D(64, (3, 3), padding='same')(x)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(x)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = Conv2D(32, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = Conv2D(256, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        h = Conv2D(512, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(h)
        h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)

        # h = MaxPooling2D((2, 2), padding='same')(h)
        h = GlobalAveragePooling2D()(h)
        # h = Flatten()(h)
        # h = Dense(self.hidden_dim/4, activation='relu')(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        lx = Lambda(sampling)([mu, log_var])

        model = Model(x, lx)
        if self.verbose:
            model.summary()

        return model

    def build_decoder(self):

        bm = 0.95

        kernel_reg_coefficient = 0.5e-4
        # kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None
        lx = Input(shape=(self.latent_dim,))

        h = Dense(self.hidden_dim / 4, activation='relu')(lx)
        # h = Dense(self.hidden_dim * 2, activation='relu')(h)
        # h = Reshape((4, 4, 16))(h)
        #
        # h = Conv2D(16, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(32, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(64, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)
        #
        # h = Conv2D(3, (3, 3), padding='same')(h)
        # h = BatchNormalization()(h)
        # h = Activation('sigmoid')(h)

        h = Dense(self.hidden_dim / 2, activation='relu', kernel_regularizer=kernel_regularizer)(h)
        h = Dense(self.hidden_dim, activation='relu', kernel_regularizer=kernel_regularizer)(h)
        h = Reshape((4, 4, 64))(h)

        h = Conv2DTranspose(512, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(512, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(256, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(256, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        # h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(128, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(128, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        # h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(64, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(64, kernel_size=4, padding='same', strides=2, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        # h = UpSampling2D((2, 2))(h)
        h = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        h = Conv2DTranspose(32, kernel_size=4, padding='same', strides=1, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        # h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('relu')(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)

        # h = UpSampling2D((2, 2))(h)

        # h = Conv2D(128, (3, 3), padding='same')(h)
        # h = BatchNormalization(momentum=bm)(h)
        # h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
        # gamma_initializer="random_uniform")(h)
        # h = Activation('relu')(h)
        # h = Conv2D(128, (3, 3), padding='same')(h)
        # h = BatchNormalization(momentum=bm)(h)
        # h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
        # gamma_initializer="random_uniform")(h)
        # h = Activation('relu')(h)
        # h = UpSampling2D((2, 2))(h)

        h = Conv2DTranspose(3, kernel_size=3, padding='same', strides=1,
                            kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(h)
        h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(h)
        # h = Conv2D(1, (3, 3), padding='same')(h)
        #h = BatchNormalization(momentum=bm)(h)
        h = Activation('tanh')(h)

        '''
        h = Conv2DTranspose(16, (3, 3), strides=(2,2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        h = Conv2D(1, (3, 3), padding='same')(h)
        #h = tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             #gamma_initializer="random_uniform")(h)
        h = Activation('relu')(h)

        '''

        model = Model(lx, h)
        if self.verbose:
            model.summary()

        return model

    def build_discriminator(self):

        bm = 0.95
        kernel_reg_coefficient = 2e-4
        # kernel_regularizer = tf.keras.regularizers.L2(kernel_reg_coefficient)
        kernel_regularizer = None
        model = Sequential()
        model.add(Dense(2048, input_dim=self.latent_dim))
        # model.add(BatchNormalization(momentum=bm))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(2048))
        # model.add(BatchNormalization(momentum=bm))
        model.add(LeakyReLU(alpha=self.alpha))
        #model.add(MinibatchDiscrimination(20, 5))

        # model.add(Dense(self.hidden_dim*2, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=self.alpha))
        # model.add(Dense(self.hidden_dim))
        # model.add(LeakyReLU(alpha=self.alpha))

        '''

        model.add(Dense(self.hidden_dim , input_dim=self.latent_dim, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Reshape((24, 24, 2)))


        model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform"))
        #model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))
        # model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(256, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))
        # model.add(BatchNormalization(momentum=bm))

        model.add(Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=True, scale=True, beta_initializer="random_uniform",
                                                   gamma_initializer="random_uniform"))

        model.add(GlobalAveragePooling2D())
        model.add(MinibatchDiscrimination(20, 5))

        '''
        '''
        model.add(Flatten())

        model.add(Dense(512, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(MinibatchDiscrimination(20, 5))

        model.add(Dense(512, kernel_regularizer = kernel_regularizer,
                            kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=self.alpha))
        '''

        model.add(Dense(1, activation='sigmoid'))
        if self.verbose:
            model.summary()

        encoded_repr = Input(shape=(self.latent_dim,))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

def main():

    # Load the dataset
    (_, _), (X_test, Y_test) = mnist.load_data()

    shape = X_test[0].shape

    input_dim = np.prod(X_test[0].shape)
    X = np.reshape(X_test, [-1, input_dim])

    latent_dim = 4
    hidden_dim = 1024
    verbose = True
    store_intermediate = True

    path = './mnist/aae/'
    name = 'mnist_aae_%d' % latent_dim

    epochs = 10000
    batch_size = 128
    sample_interval = 200

    aae = AdversarialAutoencoderMnist(shape=shape, input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                      verbose=verbose, store_intermediate=store_intermediate, path=path, name=name)

    aae.fit(X, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
    # aae.save_model()
    # aae.load_model()
    aae.sample_images(epochs)


if __name__ == '__main__':
    main()
