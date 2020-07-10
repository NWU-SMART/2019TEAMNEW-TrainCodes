# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/10 001014:02
# 文件名称：简单
# 开发工具：PyCharm
import time

import keras
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Flatten, LeakyReLU, Dropout
from keras.optimizers import Adam

path = "D:\DataList\mnist\mnist.npz"
f = np.load(path)
x_train = f['x_train']
# 归一化
x_train = x_train.astype('float32') / 255.
# 改变shape
x_train = x_train.reshape(60000, 28, 28, 1)
f.close()

D_optimizer = Adam(lr=1e-5)
G_optimizer = Adam(lr=1e-5)
# -------------生成器---------------------

G_input = Input(shape=(100,))
G = Dense(256, kernel_initializer='glorot_normal', activation='relu')(G_input)  # 初始化参数
G = BatchNormalization()(G)
# 在全连接一次
G = Dense(512, kernel_initializer='glorot_normal', activation='relu')(G)
G = BatchNormalization()(G)

G = Dense(28 * 28, kernel_initializer='glorot_normal', activation='sigmoid')(G)
G_out = Reshape((28, 28, 1))(G)  # 输出一个 28*28*1的图像
generator_model = Model(G_input, G_out)

# -------------生成器end---------------------


# ----------------辨别器-------------------

D_input = Input(shape=(28, 28, 1))  # 输入图像
D = Flatten()(D_input)
D = Dense(512)(D)
D = LeakyReLU(0.2)(D)
D = Dropout(0.25)(D)
D = Dense(256)(D)
D = LeakyReLU(0.2)(D)
D_out = Dense(1, activation='sigmoid')(D)
discriminator_model = Model(D_input, D_out)

# ----------------辨别器end-------------------


# -----------------------------编译辨别器-------#


discriminator_model.compile(loss='binary_crossentropy',  # 适用二分类
                            optimizer=D_optimizer)


# -----------------------------编译辨别器end-------#

# 让某个model不参与训练
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# --------------------定义GAN和编译GAN-----------------------
# Gan中辨别器不参与训练，Gan主要训练生成器，辨别器独立训练
make_trainable(discriminator_model, False)

Gan_input = Input(shape=[100])
img = generator_model(Gan_input)
Gan_out = discriminator_model(img)
Gan_model = Model(Gan_input, Gan_out)
Gan_model.summary()
Gan_model.compile(loss='binary_crossentropy',
                  optimizer=G_optimizer, )


# --------------------定义GAN和编译GAN--end-----------------------


# --------------------训练函数------------------------------#
def train(epochs, batch_size):
    for i in range(epochs):
        print('第{}次训练'.format(i))
        # -----------------创建辨别器的数据集-------------
        # 随机获得batch_size个真实图片
        imgs_true = x_train[np.random.randint(0, 60000, batch_size)]
        # 获得batch_size个假图像
        imgs_flase = generator_model.predict(np.random.uniform(0, 1, size=[batch_size, 100]))
        X = np.concatenate((imgs_true, imgs_flase))  # 将真实图片和虚假图片组合在一起,(里面为双括号)
        Y = np.zeros((2 * batch_size, 1))
        Y[:batch_size] = 1  # Y中前batch_size为1，表示真。后batch_size为0，表示假

        # --------------辨别器训练---------------------
        make_trainable(discriminator_model, True)  # 打开辨别器
        discriminator_model.train_on_batch(X, Y)  # 训练

        # --------------创建GAN的数据集----------------
        G_X = np.random.uniform(0, 1, size=[batch_size, 100])
        G_Y = np.ones((batch_size, 1))
        # # -----------训练GAN(即训练生成器)----------------
        make_trainable(discriminator_model, False)  # 关闭辨别器
        Gan_model.train_on_batch(G_X, G_Y)


# --------------------训练函数end------------------------------#


# -----------------开始训练--------------#
start = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
train(epochs=60000, batch_size=256)
end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
print(end)
# -----------------开始训练--------------#


import matplotlib.pyplot as plt
# ---------------生成器生成的图像-----------
x = np.random.uniform(0, 1, size=[10, 100]) # 10*100的矩阵
y_img = generator_model.predict(x)
print(discriminator_model.predict(y_img))
plt.figure(figsize=(20,6))
for i in range(10):
    # 生成器生成图像
    ax = plt.subplot(2,5,i+1)
    plt.imshow(y_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('generator.png')
plt.show()

print(time)
generator_model.save('generator.h5')

