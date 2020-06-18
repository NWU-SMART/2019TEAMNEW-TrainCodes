#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月17日
# 开发框架：keras
# 开发内容：图像生成网络（GAN）
#----------------------------------------------------------#
'''
服务器上跑的,GPU6

AttributeError: module 'keras.backend' has no attribute 'set_image_dim_ordering'
解决方法：替换为 K.set_image_data_format('channels_first')  # 改变顺序，维度放在首位
                K.image_data_format()
                “channels_first”对应原本的“th”，“channels_last”对应“tf”
'''
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, random, tqdm, IPython和 os的包
# 2、读取手图像数据及与图像预处理
# 3、超参数设置
# 4、定义生成器
# 5、定义辨别器
# 6、构造生成对抗网络
# 7、训练
# 8、输出训练数据
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
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
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # GPU
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------

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

#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------


#  --------------------- 3、超参数设置 ---------------------

# 输入、隐藏和输出层神经元个数 (1个隐藏层)

shp = X_train.shape[1:]  # 图片尺寸为 (1, 28, 28)
dropout_rate = 0.25

# Optim优化器
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)

#  --------------------- 3、超参数设置 ---------------------

#  --------------------- 4、定义生成器 ---------------------

# K.set_image_dim_ordering('th')  # 用theano的图片输入顺序
K.set_image_data_format('channels_first')  # 改变顺序，维度放在首位
K.image_data_format()

print('channels X_train shape:', X_train.shape)  # 60000*1*28*28

nch = 200

# CNN生成图片
# 通过100维的

g_input = Input(shape=[100])  # 输入100维的向量
# 100 维 --> 39200 (nch=200*14*14)， 权重 (100+1)* 39200 (input_dimendion + bias) * output_dimension
H = Dense(nch*14*14, kernel_initializer='glorot_normal')(g_input)  # Glorot正态分布初始化权重
H = BatchNormalization()(H)
H = Activation('relu')(H)

# 39200 --> 200 * 14 * 14
H = Reshape( [nch, 14, 14] )(H)  # 转成200 * 14 * 14

# 上采样 200 * 14 * 14 --> 200 * 28 * 28
H = UpSampling2D(size=(2, 2))(H)

# 200 * 28 * 28 --> 100 * 28 * 28
H = Convolution2D(100, (3, 3), padding="same", kernel_initializer='glorot_normal')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)

# 100 * 28 * 28 --> 50 * 28 * 28
H = Convolution2D(50, (3, 3), padding="same", kernel_initializer='glorot_normal')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)

# 50 * 28 * 28 --> 1 * 28 * 28
H = Convolution2D(1, (1, 1), padding="same", kernel_initializer='glorot_normal')(H)
g_V = Activation('sigmoid')(H)

# 生成generator模块
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

#  --------------------- 4、定义生成器 ---------------------


#  --------------------- 5、定义辨别器 ---------------------

# 辨别是否来自真实训练集
d_input = Input(shape=shp)

# 1 * 28 * 28 --> 256 * 14 * 14, 权重参数 (28-5+1) * 256 = 6656
H = Convolution2D(256, (5, 5), activation="relu", strides=(2, 2), padding="same")(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)

# 256 * 14 * 14 --> 512 * 7 * 7
H = Convolution2D(512, (5, 5), activation="relu", strides=(2, 2), padding="same")(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)  # 512 * 7 * 7 --> 25088   将数据“压平”,即把多维变为一维

# 25088 --> 256
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)

# 256 --> 2 (true or false)
d_V = Dense(2,activation='softmax')(H)

# 判别discriminator模块
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()


#  --------------------- 5、定义辨别器 ---------------------


#  --------------------- 6、构造生成对抗网络 ---------------------

# 冷冻训练层(定义make_trainable函数。在交替训练过程中，不需要训练辨别器，在训练生成器时)
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)

# 构造GAN
gan_input = Input(shape=[100])  # 输入数据
H = generator(gan_input)  # 生成新的图像，生成器  1 * 100 --> 1 * 28 * 28
gan_V = discriminator(H)  # 判别器  1 * 28 * 28  --> 1 * 2 (输入 [0 1]为真实图像, [1 0]为生成图像)

# 输入gan_input，生成图像，然后判别器判别输出结果为gan_V （整体GAN网络包括：生成器和判别器）
GAN = Model(gan_input, gan_V)

# GAN的Loss
GAN.compile(loss='categorical_crossentropy', optimizer=opt)

# GAN网络结果输出
GAN.summary()

#  --------------------- 6、构造生成对抗网络 ---------------------


#  --------------------- 7、训练 ---------------------

# 描绘损失收敛过程
def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


#  描绘生成器生成图像
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
    plt.savefig('gankeras.png')
    plt.tight_layout()
    plt.show()

# 抽取训练集样本 (60000个中抽取出来10000个训练)
ntrain = 10000
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
XT = X_train[trainidx,:,:,:]
print('X_train.shape', X_train.shape)  # (60000, 1, 28, 28)
print('XT.shape', XT.shape)  # (10000, 1, 28, 28)

# generator （生成器）
# discriminator  （判别器）
# GAN （generator+discriminator）  （生成器+判别器）

########### ------------------- 预训练辨别器  -----------------------------
# 预训练辨别器
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100]) # 生成XT.shape[0]个随机样本
generated_images = generator.predict(noise_gen)  # 生成器产生图片样本

# 真实图像为XT，生成图像为generated_images
X = np.concatenate((XT, generated_images))
n = XT.shape[0]

y = np.zeros([2 * n, 2])  # 构造辨别器标签 one-hot encode
y[:n, 1] = 1  # 真实图像标签 [1 0]
y[n:, 0] = 1  # 生成图像标签 [0 1]

# 使得判别器可用
make_trainable(discriminator,True)

# 预训练辨别器
discriminator.fit(X,y, epochs=1, batch_size=32)
y_hat = discriminator.predict(X)

#  计算辨别器的准确率
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_total = y.shape[0]
n_right = (diff == 0).sum()

print( "(%d of %d) right"  % (n_right, n_total))

# 存储生成器和辨别器的训练损失
losses = {"d":[], "g":[]}
########### ------------------- 预训练辨别器  -----------------------------



#  --------------------- 7、训练 ---------------------

#  --------------------- 8、输出训练数据 ---------------------

def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        ### ---- 生成器生成样本generated_images -----
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)  # generator 生成器
        ### ---- 生成器生成样本generated_images -----

        ### ------ 训练辨别器 ---------------
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discriminator, True)  # 让判别器神经网络各层可用
        d_loss = discriminator.train_on_batch(X, y)  # discriminator 判别器训练
        losses["d"].append(d_loss)  # 存储辨别器损失loss
        ### ------ 训练辨别器 ----------------

        ### ------ 训练生成对抗网络 -----------
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        # 存储生成器损失loss
        make_trainable(discriminator, False)  # 辨别器的训练关掉
        g_loss = GAN.train_on_batch(noise_tr, y2)  # GAN 生成对抗网络(包括生成器和判别器)训练
        losses["g"].append(g_loss)
        ### ------ 训练生成对抗网络 ------------


        # 更新损失loss图
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()

train_for_n(nb_epoch=1000, plt_frq=10,BATCH_SIZE=128)

#  --------------------- 8、输出训练数据 ---------------------


