# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/7 000711:12
# 文件名称：__init__.py
# 开发工具：PyCharm


# mnist图像路径
import random

import numpy

from IPython import display
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Activation, Reshape, UpSampling2D, Conv2D, Convolution2D, LeakyReLU, \
    Dropout, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

path = "D:\DataList\mnist\mnist.npz"
f = numpy.load(path)
# 得到训练集和测试集
x_train = f['x_train']
x_test = f['x_test']
f.close()

# 图像大小
img_rows, img_clos = 28, 28

# 归一化
x_train = x_train.reshape(60000, img_rows, img_clos, 1)  # 60000*28*28-->60000*28*28*1
x_test = x_test.reshape(10000, img_rows, img_clos, 1)  # 10000*28*28-->10000*28*28*1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# ----------------------超参数设置--------------------
print(x_train.shape)
shp = x_train.shape[1:]
print(shp)
dropout_rate = 0.25

# 优化器
opt = Adam(lr=0.0004)
dopt = Adam(lr=1e-5)

# ----------------------定义生成器---------------------
g_input = Input(shape=[100])  # 输入100维的向量
H = Dense(200 * 14 * 14, kernel_initializer="glorot_normal", )(g_input)  # 初始化权重(w)，输出为39200
H = BatchNormalization()(H)  # 进行批标准化
H = Activation('relu')(H)  # 使用relu激活
H = Reshape([14, 14, 200])(H)  # keras中的reshape（height,width,chns）39200-->200*14*14

H = UpSampling2D((2, 2))(H)  # 200*14*14-->200*28*28

H = Conv2D(100, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)  # 200*28*28-->100*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)

H = Conv2D(50, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)  # 100*28*28-->50*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)

H = Conv2D(1, (1, 1), padding='same', kernel_initializer="glorot_normal")(H)  # 50*28*28-->1*28*28
g_v = Activation('sigmoid')(H)

# 生成模块
generator_model = Model(g_input, g_v)
generator_model.compile(loss='binary_crossentropy', optimizer=opt)  # 优化函数为Adam(lr=0.0004)
generator_model.summary()

# --------------------定义辨别器------------------------------
d_input = Input(shape=shp)  # shape=(28,28,1) # 输入一个图像
# 28*28*1-->14*14*256,步幅为(2,2)
D = Conv2D(256, (5, 5), activation='relu', strides=(2, 2), padding='same')(d_input)
D = LeakyReLU(alpha=0.2)(D)  # LeakyReLU是给所有负值赋予一个非零斜率,x<0时，y=x*a,a是[1,+∞)，默认为0.3
D = Dropout(0.25)(D)

# 14*14*256-->7*7*512
D = Conv2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same')(D)
D = LeakyReLU(0.2)(D)
D = Dropout(0.25)(D)
# 平展，7*7*512转为25088维的向量
D = Flatten()(D)
# 25088-->256
D = Dense(256)(D)
D = LeakyReLU(0.2)(D)
D = Dropout(0.25)(D)
# 256-->2
d_v = Dense(2, activation='softmax')(D)

discriminator_model = Model(d_input, d_v)
discriminator_model.compile(loss='categorical_crossentropy', optimizer=dopt)


# ----------------------构造生成对抗网络------------------------------------
# 冷冻训练层（定义make_trainable函数，生成图像训练时,辨别图像不用训练，反之亦然）
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator_model, False)

# 构造GAN
gan_input = Input(shape=[100])  # 输入数据
H = generator_model(gan_input)  # 生成新的图像，生成器 1*100-->1*28*28
gan_v = discriminator_model(H)  # 判别器 1*28*28-->1*2,输出[0 1]为真实图像，[1 0]为生成图像

# 输入gan_input,生成图像，然后判别器判别输出结果为gan_v(整体GAN网络包括;生成器和判别器)
GAN = Model(gan_input, gan_v)

# GAN 的Loss
GAN.compile(loss='categorical_crossentropy', optimizer=opt)

# GAN 网络结果输出
GAN.summary()

# ------------------------创建数据集合，训练--------------------------

# 抽取训练集样本（60000个中抽取出来10000个训练）
ntrain = 10000
trainidx = random.sample(range(0, 60000), ntrain)  # sample 从一个list中，随机得到ntrain个list中的数字

# x_train.shape===(60000,28,28,1) XT.shape()===(10000,1,28,28,1). 也就是XT是60000中随机的10000，可以直接这样写：trainidx = random.sample(x_train, ntrain)
XT = x_train[trainidx, :, :, :]

# np.random.uniform(0,1,1200)  产生1200个[0,1)的数
noise_gen = numpy.random.uniform(0, 1, size=[10000, 100])  # size=(m,n), 则输出m*n个样本,m*n为矩阵

generator_images = generator_model.predict(noise_gen)  # 生成新图片

# 真实图像为XT，生成图像为generator_images
X = numpy.concatenate((XT, generator_images))  # 数组拼接

n = 10000
y = numpy.zeros([2 * n, 2])  # 生成20000*2全是0的矩阵，为标签，一个图像对应一个标签
y[:n, 1] = 1  # 真实图像标签[0 1]
y[n:, 0] = 1  # 生成图片标签[1 0]

# 使得判别器可用
make_trainable(discriminator_model, True)

# 预训练辨别器
# X包括10000个真图像，和10000个假图像,y为对象的标签，前10000为[0 1]后10000为[1 0]
# 判别模型开始训练
discriminator_model.fit(X, y, epochs=1, batch_size=512)
# 训练好的判别模型进行预测
y_hat = discriminator_model.predict(X)

# 计算辨别器的准确率
y_hat_idx = numpy.argmax(y_hat, axis=1)  # argmax（）对于矩阵（y_hat为矩阵），axis=1表示得到矩阵中每行最大值的索引，axis=0表示得到矩阵中每列最大值得索引
y_idx = numpy.argmax(y, axis=1)
# y_hat_idx 和 y_idx是一个向量，如果相对应索引的值不同，则不准确数加一，若索引对应的值相同，则准确数加一 如：[0 1 0 1]和[0 0 0 1]不准确数为1，准确数为3

diff = y_idx - y_hat_idx  # 向量减法，得到一个向量
n_right = (diff == 0).sum()  # diff其中为0的个数，即相同的个数
print("(%d of %d) right" % (n_right, 20000))

# 存储生成器和辨别器的训练损失
losses = {'d': [], 'g': []}


# ------------------------所用好函数-----------------------
import matplotlib.pyplot as plt


def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses['d'], label='discriminitive loss')
    plt.plot(losses['g'], label='generative loss')
    plt.legend()
    plt.show()


# 绘制生成器生成的图像
def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = numpy.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator_model.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------------输出训练数据-----------------------------------
def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):  # tqdm显示进度条
        #### --------生成器生成样本generated_images------
        # 得到随机的32个真图像
        image_batch = x_train[numpy.random.randint(0, 60000, size=BATCH_SIZE), :, :, :]

        # 得到32*100的矩阵
        noise_gen = numpy.random.uniform(0, 1, size=[BATCH_SIZE, 100])

        # 通generator_model生成图像
        generated_images = generator_model.predict(noise_gen)
        #### --------生成器生成样本generated_images------

        #### --------训练辨别器----------
        X = numpy.concatenate((image_batch, generated_images))
        y = numpy.zeros([2 * BATCH_SIZE, 2])
        y[:BATCH_SIZE, 1] = 1  # 真为[0 1]
        y[BATCH_SIZE:, 0] = 1  # 假为[1 0]

        make_trainable(discriminator_model, True)  # 打开辨别器
        d_loss = discriminator_model.train_on_batch(X, y)
        losses['d'].append(d_loss)

        noise_tr = numpy.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = numpy.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_trainable(discriminator_model, False)  # 关闭辨别器
        g_loss = GAN.train_on_batch(noise_tr, y2)  # 让生成器尽量最后生成[0 1]，即真
        losses['g'].append(g_loss)

        # 更新损失loss图
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()


train_for_n(nb_epoch=1000, plt_frq=10, BATCH_SIZE=128)

