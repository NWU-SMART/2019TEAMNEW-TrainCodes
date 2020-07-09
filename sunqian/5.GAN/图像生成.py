# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/9
# 文件名称：图像生成.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入手写图像数据并预处理
# 3、设置超参数
# 4、定义生成器-
# 5、定义辨别器
# 6、构造生成对抗网络
# 7、训练
# 8、输出训练数据
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import numpy as np
import random
from keras.layers import Input
from keras.layers.core import Dense,Reshape,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D,Deconv2D,UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *

import matplotlib.pyplot as plt
from keras.models import Model
from tqdm import tqdm # 显示循环进度条的库
from IPython import display
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入手写图像数据并预处理-------------------------------------------
# 数据集存放在本地
path = 'E:\\keras_datasets\\mnist.npz'
(X_train, Y_train), (X_test, y_test) = mnist.load_data(path)
# 观察X_train和X_test的维度
print(X_train.shape) # (60000,28,28)
print(X_test.shape) # (10000,28,28)

# 图像大小
img_rows, img_cols = 28, 28

# 数据预处理
#  归一化
X_train = X_train.reshape(60000,img_rows, img_cols,1)
X_test = X_test.reshape(10000,img_rows, img_cols,1)
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print(np.min(X_train), np.max(X_train))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 输出结果
# 0.0 1.5378702e-05
# X_train shape: (60000, 28, 28, 1)
# 60000 train samples
# 10000 test samples
#  -------------------------- 2、导入手写图像数据并预处理--------------------------------------------

#  -------------------------- 3、设置超参数 -------------------------------------------
# 输入、隐藏、输出层神经元个数(1个隐藏层)
shp = X_train.shape[1:]   # 图片尺寸是(28,28,1)
dropout_rate = 0.25
# 优化器
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)
#  -------------------------- 3、设置超参数 ------------------------------------------

#  -------------------------- 4、定义生成器-------------------------------------------
nch = 200
g_input = Input(shape=[100]) # 输入100维的向量
# 100维 -->39200(200*14*14),权重(100+1)*39200 (input_dimension+bias)*output_dimension
H = Dense(nch*14*14,kernel_initializer='glorot_normal')(g_input) # glorot_normal正态分布初始化权,输出为39200重
H = BatchNormalization()(H)  # 批标准化
H = Activation('relu')(H)

# 39200 --> 14*14*200
H = Reshape([14, 14, nch])(H)  # keras中reshape参数是(height,width,chns)

# 上采样 14*14*200 -->28*28*200
H  = UpSampling2D(size=(2, 2))(H)

# 28*28*200 --> 28*28*100
H = Convolution2D(100, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)

# 28*28*100 --> 28*28*50
H = Convolution2D(50, (3, 3), padding='same', kernel_initializer='glorot_normal')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)

# 28*28*50 --> 28*28*1
H = Convolution2D(1, (1, 1), padding='same', kernel_initializer='glorot_normal')(H)
g_V = Activation('sigmoid')(H)

# 生成generator模块
generator = Model(g_input, g_V)
generator.compile(loss='binary-crossentropy', optimizer=opt)
generator.summary()
#  -------------------------- 4、定义生成器-----------------------------------------

#  -------------------------- 5、定义辨别器-------------------------------------------
# 辨别是否来自真实训练集
d_input = Input(shape=shp)  # shape=(28,28,1) 输入一个图像

# 28*28*1 --> 14*14*256,权重参数(28-5+1)*256=6656
H = Convolution2D(256, (5, 5), activation='relu', strides=(2, 2), padding='same')(d_input)
H = LeakyReLU(0.2)(H)  # LeakyRelu激活函数是对Relu函数的改进，解决对于学习率很大的网络中较多神经元‘died’的情况
H = Dropout(dropout_rate)(H)

# 14*14*256 --> 7*7*512
H = Convolution2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)  # 将数据展平，把多维变成一维 7*7*512 -->25088

# 25088 --> 256
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)

# 256 --> 2 (true or false)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()
#  -------------------------- 5、定义辨别器------------------------------------------

#  -------------------------- 6、构造生成对抗网络 ------------------------------------------
# 冷冻训练层即定义make_trainable函数，训练生成图像时，不训练辨别图像，反之同理
def make_trainable(net, val):
    # 编译模型之前
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)

# 构造GAN
gan_input = Input(shape=[100]) # 输入的数据
H = generator(gan_input) # 生成新的图像，生成器 1*100 --> 1*28*28
gan_V = discriminator(H) # 判别器 1*28*28 --> 1*2 (输入[0,1]为真实图像，[1,0]为生成图像)
# 输入gan_input生成图像，判别器判别输出结果为gan_V
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer='opt')
GAN.summary()
#  -------------------------- 6、构造生成对抗网络-------------------------------------------

#   ---------------------------------7、训练-------------------------------------------------
# 定义函数：描绘损失收敛过程
def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.plot(losses["d"], label='dicriminitive loss')
    plt.plot(losses["g"],label='generative loss')
    plt.legend()
    plt.show()

# 定义函数：描绘生成器生成图像
def plot_gen(n_ex=16, dim=(4,4), figsize=(10,10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 抽取训练集样本（从60000中取出10000个训练）
ntrain = 10000
trainidx = random.sample(range(0, 60000), ntrain)  # sample 从一个list中随机得到ntrain个list中的数字
XT = X_train[trainidx, :, :, :]


# 预训练辨别器

# np.random.uniform(0,1,1200) 产生1200个[0,1]的数
noise_gen = np.random.uniform(0,1,size=[XT.shape[0], 100]) # XT.shape=10000 size=(m,n) 输出m*n个样本，m*n是矩阵
generated_images = generator.predict(noise_gen) # 生成器产生样本

# 真实图像是XT 生成图像是generat_images
X = np.concatenate((XT,generated_images))  # 数组拼接
n = XT.shape[0]  # x=10000
y = np.zeros([2*n, 2])  # 构造辨别器标签 独热编码 生成20000*2全为0的矩阵作为标签，一个图像对应一个标签
y[:n, 1] = 1  # 真实图像标签[0 1]
y[n:, 0] = 1  # 生成图像标签[1 0]

# 解冻判别器
make_trainable(discriminator, True)
# X包括10000个真图像和10000个生成图像,y为对象的标签，前10000为[0 1]后10000为[1 0]
# 训练判别模型
discriminator.fit(X, y, epochs=1, batch_size=32)
# 训练好的判别模型进行预测
y_hat = discriminator.predict(X)

# 计算辨别器的准确率
# argmax()对于矩阵（y_hat为矩阵），axis=1表示得到矩阵中每行最大值的索引，axis=0表示得到矩阵中每列最大值的索引
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
# y_hat_idx 和 y_idx是一个向量，如果相对应索引的值不同，则不准确数加一，若索引对应的值相同，则准确数加一 如：[0 1 0 1]和[0 0 0 1]不准确数为1，准确数为3
# 向量减法 得到一个向量
diff = y_idx-y_hat_idx
n_total = y.shape[0]
# diff其中为0的个数，即相同的个数
n_right= (diff == 0).sum()

print("(%d of %d) right" %(n_right, n_total))

# 存储生成器和辨别器的训练损失
losses = {'d':[], 'g':[]}
# ----------输出训练数据--------
def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):  # tqdm()显示进度条

        # 生成器生成样本
        # 得到随机的32个真图像
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:, :, :]

        # 得到32*100的矩阵
        noise_gen= np.random.uniform(0, 1, size=[BATCH_SIZE,100])
        # 通过generator.predict生成图像
        generated_images =generator.predict(noise_gen)


        #  训练辨别器
        X = np.concatenate((image_batch,generated_images))
        y = np.zeros([2*BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1  # 真为[0 1]
        y[BATCH_SIZE:, 0] = 1   # 假为[1 0]

        # 存储辨别器损失loss
        make_trainable(discriminator,True)  # 打开辨别器的训练
        d_loss = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # 生成器生成样本
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:, 1] = 1

        # 存储生成器损失loss
        make_trainable(discriminator,False)  # 关闭辨别器的训练
        g_loss = GAN.train_on_batch(noise_tr, y2)  # 让生成器尽量最后生成[0 1],即真
        losses["g"].append(g_loss)

        # 更新损失loss图
        if e%plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()

train_for_n(nb_epoch=100,plt_frq=10,BATCH_SIZE=32)


#   ---------------------------------7、训练--------------------------------------------------