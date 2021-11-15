# -*- coding: utf-8 -*-
"""Networks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18JhksEg84TbJM-zyNO1zbdK_GzNnt8Q1
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Mizani implementation/

from modules import gated_conv2d, IGRB, CSAB, SPD, SPD_4, Self_attention
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

### Fine Encoder
def Fine_encoder(input_shape, latent_dim = 2):
  input_1 = Input(shape = input_shape)
  
  x = gated_conv2d(input_1,filters=32,kernel_size=(3,3),padding='same')

  x = gated_conv2d(x,filters=64,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=128,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same')
  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same')
  x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')
  x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')
  f = x
  x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')
  x = GlobalAveragePooling2D()(x)
  x = Dense(16,activation='relu')(x)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(inputs=input_1,outputs=[z_mean, z_log_var, z, f], name='fine_Encoder')
  return encoder

### Coarse Encoder
def Coarse_encoder(input_shape, latent_dim = 2):
  input_1 = Input(shape = input_shape)

  x = Conv2D(filters=32,kernel_size=(3,3),padding='same')(input_1)
  x = Activation('relu')(x)

  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = Activation('relu')(x)

  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = Activation('relu')(x)

  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = Activation('relu')(x)

  x = SPD(x, kernel_size=(3,3), filters=256)
  x = SPD(x, kernel_size=(3,3), filters=256)
  x = SPD(x, kernel_size=(3,3), filters=256)
  x = GlobalAveragePooling2D()(x)
  x = Dense(16,activation='relu')(x)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(inputs=input_1,outputs=[z_mean, z_log_var, z], name='coarse_Encoder')
  return encoder

def Decoder(latent_dim=2):
  latent_input1 = Input(shape=(latent_dim,))
  latent_input2 = Input(shape=(latent_dim,))
  f_input = Input(shape=(32,32,256))

  latent1 = Dense(32*32*64, activation='relu')(latent_input1)
  latent1 = Reshape((32,32,64))(latent1)

  latent2 = Dense(32*32*64, activation='relu')(latent_input2)
  latent2 = Reshape((32,32,64))(latent2)

  x = Concatenate()([latent1,latent2,f_input])
  # x = Concatenate()(x,f_input)

  x = CSAB(x)

  x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2),activation='relu',padding="same")(x)
  x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(1,1),activation='relu', padding="same")(x)
  x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(2,2),activation='relu', padding="same")(x)
  x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu', padding="same")(x)
  x = Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2),activation='relu', padding="same")(x)
  x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(1,1),activation='sigmoid', padding="same")(x)
  decoder = keras.Model(inputs=[latent_input1,latent_input2,f_input],outputs = x)

  return decoder

from sn import SpectralNormalization

def Discriminator (input_shape=(256,256,3)):
  input_1 = Input(shape=input_shape)

  x = SpectralNormalization(Conv2D(filters = 64,kernel_size=(4,4),strides=(2,2),padding='same'))(input_1)
  x = LeakyReLU()(x)

  x = SpectralNormalization(Conv2D(filters = 128,kernel_size=(4,4),strides=(2,2),padding='same'))(x)
  x = LeakyReLU()(x)  

  x = SpectralNormalization(Conv2D(filters = 256,kernel_size=(4,4),strides=(2,2),padding='same'))(x)
  x = LeakyReLU()(x)

  x = SpectralNormalization(Conv2D(filters = 512,kernel_size=(4,4),strides=(1,1),padding='same'))(x)
  x = LeakyReLU()(x)

  x = SpectralNormalization(Conv2D(filters = 1,kernel_size=(4,4),strides=(2,2),padding='same'))(x)
  x = Activation('sigmoid')(x)

  discriminator = keras.Model(inputs=input_1,outputs=x)
  return discriminator

### Fine Encoder
def Fine_encoder_g(input_shape):
  input_1 = Input(shape = input_shape)
  
  x = gated_conv2d(input_1,filters=32,kernel_size=(3,3),padding='same')

  x = gated_conv2d(x,filters=64,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=128,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same',strides=(2,2))

  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same')
  x = gated_conv2d(x,filters=256,kernel_size=(3,3),padding='same')
  x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')
  x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')

  # x = IGRB(x,filters=256,kernel_size=(3,3),padding='same')

  encoder = keras.Model(inputs=input_1,outputs= x, name='fine_Encoder_g')
  return encoder

### Coarse Encoder
def Coarse_encoder_g(input_shape):
  input_1 = Input(shape = input_shape)

  x = Conv2D(filters=32,kernel_size=(3,3),padding='same')(input_1)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)

  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)

  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)

  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Activation('relu')(x)
  x = LeakyReLU(0.3)(x)
  x = SPD(x, kernel_size=(3,3), filters=256)
  x = SPD(x, kernel_size=(3,3), filters=256)
  x = SPD(x, kernel_size=(3,3), filters=256)
  x = SPD(x, kernel_size=(3,3), filters=256)

  encoder = keras.Model(inputs=input_1,outputs= x, name='coarse_Encoder_g')
  return encoder

def Decoder_g(input_shape = (32,32,256)):

  f_input1 = Input(shape=input_shape)
  # f_input2 = Input(shape=input_shape)

  x = CSAB(f_input1)

  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # # x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(1,1), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # # x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(1,1), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=3,kernel_size=(3,3),padding='same',activation='sigmoid',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(1,1),activation='sigmoid', padding="same")(x)

  decoder = keras.Model(inputs=f_input1,outputs = x)

  return decoder

def Decoder_g_natt(input_shape = (32,32,256)):

  f_input1 = Input(shape=input_shape)
  # f_input2 = Input(shape=input_shape)

  # x = CSAB(f_input1)

  x = UpSampling2D(size=(2,2), interpolation='bilinear')(f_input1)
  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # # x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(1,1), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # # x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(1,1), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2), padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=3,kernel_size=(3,3),padding='same',activation='sigmoid',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(1,1),activation='sigmoid', padding="same")(x)

  decoder = keras.Model(inputs=f_input1,outputs = x)

  return decoder

def refinement_network(input_shape = (256,256,3)):

  input_1 = Input(shape = input_shape)

  x = Conv2D(filters=32,kernel_size=(3,3),padding='same')(input_1)
  x = LeakyReLU(0.3)(x)

  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = LeakyReLU(0.3)(x)

  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = LeakyReLU(0.3)(x)

  f1 = x

  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = LeakyReLU(0.3)(x)

  f2 = x

  x = Conv2D(filters=512,kernel_size=(3,3),padding='same',strides=(2,2))(x)
  x = LeakyReLU(0.3)(x)

  x = SPD_4(x, kernel_size=(3,3), filters=512)
  x = SPD_4(x, kernel_size=(3,3), filters=512)
  x = SPD_4(x, kernel_size=(3,3), filters=512)

  x = Self_attention(x)

  x = SPD_4(x, kernel_size=(3,3), filters=512)
  x = SPD_4(x, kernel_size=(3,3), filters=512)
  x = SPD_4(x, kernel_size=(3,3), filters=512)

  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=256,kernel_size=(3,3),strides=(2,2),padding="same")(x)
  x = LeakyReLU(0.3)(x)

  f2 = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(f2)
  f2 = LeakyReLU(0.3)(f2)
  f2 = Self_attention(f2)

  x = Concatenate()([x,f2])
  
  x = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  x = LeakyReLU(0.3)(x)

  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2),padding="same")(x)
  x = LeakyReLU(0.3)(x)

  f1 = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(f1)
  f1 = LeakyReLU(0.3)(f1)
  f1 = Self_attention(f1)

  x = Concatenate()([x,f1])

  x = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  x = LeakyReLU(0.3)(x)

  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=(2,2),padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
  x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1))(x)
  # x = Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2),padding="same")(x)
  x = LeakyReLU(0.3)(x)
  x = Conv2D(filters=3,kernel_size=(3,3),activation='sigmoid',padding='same',strides=(1,1))(x)

  # x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(1,1),activation='relu',padding="same")(x)

  network = keras.Model(inputs=input_1,outputs = x)

  return network