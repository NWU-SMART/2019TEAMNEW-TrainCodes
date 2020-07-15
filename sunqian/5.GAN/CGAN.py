# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/15
# 文件名称：CGAN.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入手写图像数据并预处理
# 3、设置超参数
# 4、定义生成器
# 5、定义辨别器
# 6、构造生成对抗网络
# 7、训练
# 8、输出训练数据
# ----------------------   代码布局： ----------------------
'''
CGAN一种带条件约束的GAN，在生成模型（G）和判别模型（D）的建模中均引入条件变量y（conditional variable y）
使用额外信息y对模型增加条件，可以指导数据生成过程。这些条件变量y可以基于多种信息，例如类别标签，用于图像修复的部分数据，来自不同模态（modality）的数据
如果条件变量y是类别标签，可以看做CGAN是把纯无监督的 GAN 变成有监督的模型的一种改进
普通的GAN输入的是一个N维的正态分布随机数，而CGAN会为这个随机数添上标签，
其利用Embedding层将正整数（索引值）转换为固定尺寸的稠密向量，并将这个稠密向量与N维的正态分布随机数相乘，从而获得一个有标签的随机数。
'''
#  -------------------------- 1、导入需要的包 -------------------------------
import time
from keras import models
from keras.datasets import mnist
import numpy as np
from keras.layers import Input,Dense,Reshape,Dropout,Activation,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D,Embedding,multiply,UpSampling2D
from keras.layers.normalization import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入手写图像数据并预处理-------------------------------------------
# 数据集存放在本地
path = 'E:\\keras_datasets\\mnist.npz'
(X_train, y_train), (_, _) = mnist.load_data(path)
# 观察X_train训练数据和y_train训练标签的维度
print(X_train.shape)  # (60000,28,28)
print(y_train.shape)  # (60000,)

# 图像大小
img_rows, img_cols = 28, 28

# 数据预处理
#  归一化
X_train = X_train.reshape(60000, img_rows, img_cols, 1)
X_train = X_train.astype("float32")/255.
y_train = y_train.reshape(60000, 1)

print(np.min(X_train), np.max(X_train))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(y_train.shape[0], 'train labels')

# 输出结果
# 0.0 1.0
# X_train shape: (60000, 28, 28, 1)
# 60000 train samples
# 60000 train labels
#  -------------------------- 2、导入手写图像数据并预处理--------------------------------------------

#  -------------------------- 3、设置超参数 -------------------------------------------
# 输入、隐藏、输出层神经元个数(1个隐藏层)
dropout_rate = 0.25
# 优化器
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)
#  -------------------------- 3、设置超参数 ------------------------------------------

#  -------------------------- 4、定义生成器-------------------------------------------
# 生成网络的输入时一个带标签的随机数，具体操作方式是生成一个N维的正态分布随机数，再利用Embedding层将
# 正整数(索引值)转换为N维的稠密向量，并将这个稠密向量与N维的正态分布随机数相乘
nch = 200
g_input = Input(shape=[100])  # 输入100维的向量

# -------------------处理输入标签--------------------
# 输入标签
label_input = Input(shape=(1,), dtype='int32')
# 经过embedding转为1*100的矩阵，10表示label中最大数加一，即最大为9(0-9),加一变为10，输出维度为100
L= Embedding(10,100)(label_input)
# 展开为100维的向量
label = Flatten()(L)
# 将输入的100维随机向量乘以输入的标签（也是100维） multiply两个向量相乘运算 multiply([1 2 3],[4 5 6])=[4 10 18]
model_input = multiply([g_input, label])
# -------------------处理输入标签--------------------

# 100维 -->39200(200*14*14),权重(100+1)*39200 (input_dimension+bias)*output_dimension
G = Dense(nch*14*14, kernel_initializer='glorot_normal')(model_input) # glorot_normal正态分布初始化权,输出为39200
G = BatchNormalization()(G)  # 批标准化
G = Activation('relu')(G)

# 39200 --> 14*14*200
G = Reshape([14, 14, nch])(G)  # keras中reshape参数是(height,width,channels)

# 上采样 14*14*200 -->28*28*200  size=(2,2)沿着数据的行和列分别重复 size[0] 和 size[1] 次
G = UpSampling2D(size=(2, 2))(G)

# 28*28*200 --> 28*28*100  步长元组strides=(3,3)
G = Conv2D(100, (3, 3), padding='same', kernel_initializer='glorot_normal')(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)

# 28*28*100 --> 28*28*50
G = Conv2D(50, (3, 3), padding='same', kernel_initializer='glorot_normal')(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)

# 28*28*50 --> 28*28*1
G = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_normal')(G)
g_output = Activation('sigmoid')(G)

# 生成generator模块
generator = Model([g_input, label_input], g_output)
# generator.compile(loss='binary-crossentropy', optimizer=opt)  # 二元交叉熵损失函数
generator.summary()
#  -------------------------- 4、定义生成器-----------------------------------------

#  -------------------------- 5、定义辨别器-------------------------------------------
# 辨别是否来自真实训练集
d_input = Input(shape=(28, 28, 1))  # shape=(28,28,1) 输入一个图像

# 28*28*1 --> 14*14*256,权重参数(28-5+1)*256=6656
D = Conv2D(256, (5, 5), activation='relu', strides=(2, 2), padding='same')(d_input)
D = LeakyReLU(0.2)(D)  # LeakyRelu激活函数是对Relu函数的改进，解决对于学习率很大的网络中较多神经元‘died’的情况
D = Dropout(dropout_rate)(D)

# 14*14*256 --> 7*7*512
D = Conv2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same')(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
D = Flatten()(D)  # 将数据展平，把多维变成一维 7*7*512 -->25088

# 25088 --> 256
D = Dense(256)(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)


# 两个输出，一个判断输出的真假（二分类），一个判断输出类别向量（多分类）
d_validity = Dense(1, activation='sigmoid')(D)
d_label = Dense(10, activation='softmax')(D)

discriminator = Model(d_input,[d_validity,d_label])
# 编译辨别器
# 由于辨别器有两个输出，所以需要两个损失函数,分别适用于二分类和多分类输出
# categorical_crossentropy主要用于独热编码 如[1,0,0] [0,1,0] [0,0,1]
# sparse_categorical_crossentropy用于目标结果是整个整数，如1,2,3
losses = ['categorical_crossentropy', 'sparse_categorical_crossentropy']
discriminator.compile(loss=losses, optimizer=dopt,metrics=['acc'])
discriminator.summary()
#  -------------------------- 5、定义辨别器------------------------------------------

#  -------------------------- 6、构造生成对抗网络 ------------------------------------------
# 构造GAN
# GAN 将生成器和辨别器组合，即两个输入 两个输出
gan_input1 = Input(shape=(100,))  # 输入的数据100维的向量
gan_input2 = Input(shape=(1,))  # 输入要生成的数字
generated_image = generator([gan_input1, gan_input2])  # 生成新的图像

# 得到两个输出，一个判别真假，一个判别标签
gan_output1, gan_output2 = discriminator(generated_image)  # 判别器 1*28*28 --> 1*2 (输入[0,1]为真实图像，[1,0]为生成图像)
# 输入gan_input生成图像，判别器判别输出结果为gan_V
GAN_model = Model([gan_input1, gan_input2], [gan_output1,gan_output2])
# 编译GAN
GAN_model.compile(loss=losses, optimizer=dopt)
GAN_model.summary()
#  -------------------------- 6、构造生成对抗网络-------------------------------------------

#   ---------------------------------7、训练GAN和辨别器-------------------------------------------------
# 冷冻训练层即定义make_trainable函数，训练生成图像时，不训练辨别图像，反之同理
def make_trainable(net, val):
    # 编译模型之前
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def train(epochs, batch_size):
    y_img_true = np.ones((batch_size, 1))  # 真实图像为1
    y_img_false = np.zeros((batch_size, 1))  # 生成图像为0
    for i in range(epochs):
        print('第{}次训练'.format(i))

        # -----------------构建辨别器训练集------------
        # 辨别器需要一个输入(图像)和两个输出(真假，标签)
        idx = np.random.randint(0, 60000, batch_size)
        # 真实图像和标签
        x_img_true = X_train[idx]
        y_label_true = y_train[idx]
        # 生成图像和标签由生成器生成，生成器需要两个输入 input_false y_label_false
        input_false = np.random.normal(0, 1, (batch_size, 100))  # size是batch_size*100的矩阵
        y_label_false = np.random.randint(0, 10, (batch_size, 1))  # size是batch_size*1的矩阵
        # 得到生成图像
        x_img_false = generator.predict([input_false, y_label_false])

        # X由真实图像和生成图像构成，前面为真，后面为生成（假）
        X = np.concatenate([x_img_true, x_img_false])
        # Y判断真假，前面为真，后面为生成(假)
        Y = np.concatenate([y_img_true, y_img_false])
        # Z判断是标签几
        Z = np.concatenate([y_label_true, y_label_false])
        # -----------------构建辨别器训练集------------
        # -----------------训练辨别器------------
        make_trainable(discriminator, True)
        discriminator.train_on_batch(X, [Y, Z])
        # -----------------训练辨别器------------
        # -----------------训练GAN------------
        make_trainable(discriminator,False)
        GAN_model.train_on_batch([input_false, y_label_false], [y_img_true, y_label_false])
        # -----------------训练GAN------------
        if i == 100:
            generator.save('generator_model_1.h5')
        if i == 150:
            generator.save('generator_model_1.5.h5')
        if i == 200:
            generator.save('generator_model_2.h5')

# 显示训练时间
start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(start)

train(102, 256)

end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(end)
#   ---------------------------------7、训练GAN和辨别器--------------------------------------------------


# -----------------------------------8、查看训练效果-----------------------------------
x = np.random.uniform(0, 1, size=[10, 100])  # 10*100的矩阵
y = np.array([0,1,2,3,4,5,6,7,8,9])
generator_model = models.load_model('generator_model_1.h5')
y_img = generator_model.predict([x, y])
# 画0-9手写数字的图像
plt.figure(figsize=(20, 6))
for i in range(10):
    # 生成器生成图像
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(y_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('generator_mnist.png')
plt.show()
# -----------------------------------8、查看训练效果-----------------------------------