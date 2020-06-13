# -*- coding: utf-8 -*-
# @Time: 2020/6/10 10:19
# @Author: wangshengkang
# ----------------------------------------代码布局--------------------------------------------
# 1导入包
# 2导入数据，图像预处理
# 3超参数设置
# 4构建生成器模型
# 5构建判别器模型
# 6构建gan模型
# 7训练
# 8输出训练数据
# ------------------------------------------1导入包---------------------------------------------
import random
import numpy as np
from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
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
from keras import backend as K
import os

# ------------------------------------------1导入包---------------------------------------------
# ------------------------------------------2导入数据，数据预处理-------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 选择gpu
plt.switch_backend('agg')  # 服务器没有gui
path = 'mnist.npz'
f = np.load(path)  # 导入数据
X_train = f['x_train']  # 训练集
X_test = f['x_test']  # 测试集
f.close()
print(X_train.shape)  # 60000*28*28

img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  # 增加通道维度
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32") / 255.  # 归一化
X_test = X_test.astype("float32") / 255.
print(X_train.shape)  # 60000*1*28*28

# ------------------------------------------2导入数据，数据预处理-------------------------------------------
# ------------------------------------------3超参数设置-------------------------------------------
shp = X_train.shape[1:]  # 1*28*28
print('shp', shp)
dropout_rate = 0.25

opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)

K.set_image_data_format('channels_first')  # 改变顺序，维度放在首位
K.image_data_format()

print('channels X_train shape:', X_train.shape)  # 60000*1*28*28

nch = 200
# ------------------------------------------3超参数设置-------------------------------------------
# ------------------------------------------4定义生成器-------------------------------------------
g_input = Input(shape=[100])  # 输入100维的向量
H = Dense(nch * 14 * 14, kernel_initializer='glorot_normal')(g_input)  # 39200，Glorot正态分布初始化权重
H = BatchNormalization()(H)
H = Activation('relu')(H)

H = Reshape([nch, 14, 14])(H)  # 200*14*14

H = UpSampling2D(size=(2, 2))(H)  # 200*28*28

H = Convolution2D(100, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)  # 100*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)

H = Convolution2D(50, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)  # 50*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)

H = Convolution2D(1, (1, 1), padding='same', kernel_initializer='glorot_normal')(H)  # 1*28*28
g_v = Activation('sigmoid')(H)

generator = Model(g_input, g_v)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

# ------------------------------------------4定义生成器-------------------------------------------
# ------------------------------------------5定义判别器-------------------------------------------
d_input = Input(shape=shp)  # 1*28*28

H = Convolution2D(256, (5, 5), activation='relu', strides=(2, 2), padding='same')(d_input)  # 256*14*14
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)

H = Convolution2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same')(H)  # 512*7*7
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)  # 25088

H = Dense(256)(H)  # 256
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)

d_V = Dense(2, activation='softmax')(H)  # 2，真或假

discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()


# ------------------------------------------5定义判别器-------------------------------------------
# ------------------------------------------6构造生成对抗网络-------------------------------------------

# 由于gan的交替训练机制，训练生成器时，不需要训练判别器
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator, False)  # 不训练判别器

gan_input = Input(shape=[100])  # 输入100维的数据       100
H = generator(gan_input)  # 生成器，生成图片            1*28*28
gan_V = discriminator(H)  # 判别器，进行判别            2

GAN = Model(gan_input, gan_V)  # gan完整的网络模型

GAN.compile(loss='categorical_crossentropy', optimizer=opt)

GAN.summary()


# ------------------------------------------6构造生成对抗网络-------------------------------------------
# ------------------------------------------7训练-------------------------------------------------------
# 画损失函数的曲线图
def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses['d'], label='discriminitive loss')
    plt.plot(losses['g'], label='generative loss')
    plt.legend()
    plt.show()


# 画生成的gan的图片
def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    print('save picture--------------------------------------------------------')
    plt.savefig('51gankeras.png')
    plt.tight_layout()
    plt.show()


# 预训练判别器
ntrain = 10000  # 从训练集60000个样本中抽取10000个
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)  # 随机抽取
XT = X_train[trainidx, :, :, :]
print('X_train.shape', X_train.shape)  # (60000, 1, 28, 28)
print('XT.shape', XT.shape)  # (10000, 1, 28, 28)

noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])  # 生成10000个随机样本
generated_images = generator.predict(noise_gen)  # 生成器根据随机样本生成图片

X = np.concatenate((XT, generated_images))  # XT为真实图像，generated_images为生成图像
n = XT.shape[0]

y = np.zeros([2 * n, 2])  # 构造判别器标签，one-hot编码
y[:n, 1] = 1  # 真实图像标签[1 0]
y[n:, 0] = 1  # 生成图像标签[0 1]

make_trainable(discriminator, True)  # 训练判别器

discriminator.fit(X, y, epochs=1, batch_size=32)  # 预训练判别器
y_hat = discriminator.predict(X)

# 计算判别器准确率
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_total = y.shape[0]
n_right = (diff == 0).sum()
print('(%d of %d) right' % (n_right, n_total))

# 存储生成器和判别器的训练损失
losses = {'d': [], 'g': []}


# ------------------------------------------7训练-------------------------------------------------------
# ------------------------------------------8输出训练数据-----------------------------------------------

def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        # 生成器生成样本
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)

        # 训练判别器
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(X, y)
        losses['d'].append(d_loss)

        # 训练生成对抗网络
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_trainable(discriminator, False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses['g'].append(g_loss)

        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()


train_for_n(nb_epoch=1000, plt_frq=10, BATCH_SIZE=128)
# ------------------------------------------8输出训练数据-----------------------------------------------
