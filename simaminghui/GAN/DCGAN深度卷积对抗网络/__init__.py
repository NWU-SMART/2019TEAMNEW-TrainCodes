# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/11 001111:21
# 文件名称：__init__.py
# 开发工具：PyCharm
import time

import numpy as np
import keras

path = "D:\DataList\mnist\mnist.npz"
f = np.load(path)
x_train = f['x_train']

# 数据处理
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
f.close()


# ----------------------生成器-------------------------#
def generator():
    G_input = keras.Input(shape=(100,))
    G = keras.layers.Dense(7 * 7 * 32, activation='relu')(G_input)  # 全连接层
    G = keras.layers.BatchNormalization()(G)  # 批标准化
    # 7*7*32
    G = keras.layers.Reshape((7, 7, 32))(G)

    # 进入卷积，7*7*32-->7*7*64
    G = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(G)
    G = keras.layers.BatchNormalization()(G)  # 标量标准化

    # 上采样 7*7*64-->14*14*64
    G = keras.layers.UpSampling2D()(G)  # 上采样
    # 卷积 14*14*64-->14*14*128
    G = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(G)
    G = keras.layers.BatchNormalization()(G)  # 标量标准化

    # 上采样 14*14*128-->28*28*128
    G = keras.layers.UpSampling2D()(G)
    # 卷积 28*28*128-->28*28*64
    G = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(G)
    G = keras.layers.BatchNormalization()(G)

    # 28*28*64-->28*28*1
    G_out = keras.layers.Conv2D(1, kernel_size=3, padding='same', activation="sigmoid")(G)

    return keras.Model(G_input, G_out)


# ----------------------生成器end------------------------#


# ----------------------辨别器-------------------------#
def discriminator():
    D_input = keras.Input(shape=(28, 28, 1))
    # 28*28*1-->14*14*32
    D = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(D_input)
    D = keras.layers.LeakyReLU(0.2)(D)
    D = keras.layers.Dropout(0.25)(D)

    # 14*14*32-->7*7*64
    D = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(D)
    D = keras.layers.LeakyReLU(0.2)(D)
    D = keras.layers.Dropout(0.25)(D)

    # 7*7*64-->4*4*128
    D = keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(D)
    D = keras.layers.LeakyReLU(0.2)(D)
    D = keras.layers.Dropout(0.25)(D)

    # 进入全连接层
    D = keras.layers.Flatten()(D)
    D_out = keras.layers.Dense(1, activation='sigmoid')(D)

    return keras.Model(D_input, D_out)


# ----------------------辨别器end-------------------------#


# ----------------------生成生成器和辨别器model-------------------------#
generator_model = generator()
discriminator_model = discriminator()
# ----------------------生成生成器和辨别器model-end------------------------#


# ----------------------编译辨别器-------------------------#
opt = keras.optimizers.Adam(lr=0.0004)
discriminator_model.compile(loss='binary_crossentropy',  # 二分类
                            optimizer=opt
                            )


# ----------------------编译辨别器end-------------------------#


# ----------------------是否训练辨别器-------------------------#
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# ----------------------是否训练辨别器end-------------------------#


# ----------------------定义GAN，编译GAN-------------------------#
gopt = keras.optimizers.Adam(lr=0.0004)
GAN_input = keras.Input(shape=(100,))
img = generator_model(GAN_input)
GAN_out = discriminator_model(img)
GAN_model = keras.Model(GAN_input, GAN_out)
GAN_model.compile(loss='binary_crossentropy',
                  optimizer=gopt)


# ----------------------定义GAN，编译GAN-end------------------------#


# ----------------------训练函数-------------------------#
def train(epochs, size):
    for i in range(epochs):
        print("第{}次训练".format(i))
        # ------------- 构建辨别器训练集--------------#
        # 得到随机的size个真图像
        imgs_true = x_train[np.random.randint(0, 60000, size)]
        # 得到size个假图像
        imgs_false = generator_model.predict(np.random.normal(0, 1, size=[size, 100]))
        X = np.concatenate((imgs_true, imgs_false))
        Y = np.zeros((2 * size, 1))
        # 前size为真，值为1，后size为假，为0
        Y[:size] = 1
        # ------------- 构建辨别器训练集end--------------#

        # --------------辨别器训练----------------#
        make_trainable(discriminator_model, True)
        discriminator_model.train_on_batch(X, Y)
        # --------------辨别器训练end----------------#

        # --------------构建GAN训练数据集----------#
        G_X = np.random.normal(0, 1, size=[size, 100])
        G_Y = np.ones((size, 1))
        # --------------构建GAN训练数据集end----------#

        # --------------训练GAN----------#
        make_trainable(discriminator_model, False)
        GAN_model.train_on_batch(G_X, G_Y)
        # --------------训练GAN-end---------#

        if i == 500:
            generator_model.save('generator_model_0.5.h5')
        if i == 1000:
            generator_model.save('generator_model_1.h5')
        if i == 1500:
            generator_model.save('generator_model_1.5.h5')
        if i == 2000:
            generator_model.save('generator_model_2.h5')





# ----------------------训练函数end-------------------------#
start = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
train(2000, 128)
end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
print(end)
