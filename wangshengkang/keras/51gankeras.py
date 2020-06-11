# -*- coding: utf-8 -*-
# @Time: 2020/6/10 10:19
# @Author: wangshengkang

import random
import numpy as np
from keras.layers import Input
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Model
from tqdm import tqdm
from IPython import display

path='mnist.npz'
f=np.load(path)
X_train=f['x_train']
X_test=f['x_test']
f.close()

img_rows,img_cols=28,28

X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
X_train=X_train.astype("float32")/255.
X_test=X_test.astype("float32")/255.

shp=X_train.shape[1:]
dropout_rate=0.25

opt=Adam(lr=1e-4)
dopt=Adam(lr=1e-5)

K.image_data_format()=='channels_first'
nch=200
g_input=Input(shape=[100])
H=Dense(nch*14*14,kernel_initializer='glorto_normal')(g_input)
H=BatchNormalization()(H)
H=Activation('relu')(H)

H=Reshape([nch,14,14])(H)

H=UpSampling2D(size=(2,2))(H)

H=Convolution2D(100,(3,3),padding='same',kernel_initializer='glorot_normal')(H)
H=BatchNormalization()(H)
H=Activation('relu')(H)

H=Convolution2D(50,(3,3),padding='same',kernel_initializer='glorot_normal')(H)
H=BatchNormalization()(H)
H=Activation('relu')(H)

H=Convolution2D(1,(1,1),padding='same',kernel_initializer='golrot_normal')(H)
g_v=Activation('sigmoid')(H)

generator=Model(g_input,g_v)
generator.compile(loss='binary_crossentropy',optimizer=opt)
generator.summary()