
from modules import gated_conv2d, IGRB, CSAB, SPD, SPD_4, Self_attention
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


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
