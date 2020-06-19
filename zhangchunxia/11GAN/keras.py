# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月19日
# 内容:图像生成-GAN
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手图像数据及与图像预处理
# 3、超参数设置
# 4、定义生成器
# 5、定义辨别器
# 6、构造生成对抗网络
# 7、训练
# 8、输出训练数据
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import random            #随机模块，在一个范围个抽取每个数的概率是一样的
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
from tqdm import tqdm        #它的作用就是在终端上出现一个进度条，使得代码进度可视化
from IPython import display
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'
f = np.load(path)
X_train=f['x_train']
X_test=f['x_test']
f.close()
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)
img_rows, img_cols = 28, 28#输入图像的维度
#数据预处理，归一化
X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
#数据类型转换和归一化，将像素点转化到[0，1]之间
X_train = X_train.astype("float32")/255.#数据转换为float类型
X_test = X_test.astype("float32")/255.
print(np.min(X_train),np.max(X_train))
print('X_train.shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
#  ---------------------- 3、超参数设置 -----------------------------------
#输入，隐藏和输出层，隐藏层只有一层
shp = X_train.shape[1:]#(1*28*28为图片尺寸）
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)
#  ---------------------- 3、超参数设置 -----------------------------------
#  ---------------------- 4、定义生成器 -----------------------------------
#K.set_image_dim_ordering('th')
K.image_data_format() == 'channels_first'#用 K.image_data_format() == 'channels_first' 替换K.image_dim_ordering() == 'th'成功解决
nch = 200
#CNN生成图片
g_input= Input(shape=[100])
#100 -39200
H = Dense(nch*14*14, kernel_initializer='glorot_normal')(g_input)#初始化方法定义了对Keras层设置初始化权重的方法,正态分布初始化方法
H = BatchNormalization()(H)#在每一层输入的时候，加入归一化层，先做一个归一化处理，再到网络的下一层
H = Activation('relu')(H)
#39200-200*14*14
H = Reshape([nch,14,14])(H)
#39200-200*28*28
H = UpSampling2D(size=(2,2))(H)
H = Convolution2D(100,(3,3),padding='same',kernel_initializer='glorot_normal')(H)#卷积层，200*28*28-100*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(50,(3,3),padding='same',kernel_initializer='glorot_normal')(H)#卷积层，100*28*28-50*28*28
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(1,(1,1),padding='same',kernel_initializer='glorot_normal')(H)#卷积层，50*28*28-1*28*28
g_v = Activation('sigmoid')(H)
#生成器模块
generate = Model(g_input,g_v)
generate.compile(loss='binary_crossentropy',optimizer=opt)
generate.summary()
#  ---------------------- 4、定义生成器 -----------------------------------
#  ---------------------- 5、定义鉴别器 -----------------------------------
d_input = Input(shape=shp)
# 1 * 28 * 28 --> 256 * 14 * 14, 权重参数 (28-5+1) * 256 = 6656
H = Convolution2D(256,(5,5),activation='relu',strides=(2,2),padding='same')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
# 256 * 14 * 14 --> 512 * 7 * 7
H = Convolution2D(512,(5,5),activation='relu',strides=(2,2),padding='same')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)#将数据铺平，多维转换为一维，512 * 7 * 7 --> 25088
# 25088 --> 256
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_v = Dense(2,activation='softmax')(H)
#鉴别器模块
discriminator = Model(d_input,d_v)
discriminator.compile(loss='categorical_crossentropy',optimizer=dopt)
discriminator.summary()
#  ---------------------- 5、定义鉴别器 -----------------------------------
#  ---------------------- 6、构造生成对抗网络 -----------------------------
# 冷冻训练层(定义make_trainable函数。在交替训练过程中，不需要训练辨别器)
def make_trainable(net,val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)
#构建GAN
gan_input = Input(shape=[100])  #输入的数据
H = generate(gan_input)         #生成新的图像，生成器1*100---1 * 28 * 28
gan_v = discriminator(H)        # 判别器  1 * 28 * 28  --> 1 * 2 (输入 [0 1]为真实图像, [1 0]为生成图像)
GAN = Model(gan_input,gan_v)    #鉴别器判别生成的图像
GAN.compile(loss='categorical_crossentropy',optimizer=opt)
GAN.summary()#GAN结果输出
#  ---------------------- 6、构造生成对抗网络 -----------------------------
#  ---------------------- 7、训练 -----------------------------------------
def plot_loss(losses):         #描绘损失收敛
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"],label='discriminiative loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()
def plot_gen(n_ex=16,dim=(4,4),figsize=(10,10)):#描绘生成器生成图像
    noise = np.randon.uniform(0,1,size=[n_ex,100])
    generated_images = generate.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
#从60000个训练样本中随机抽取10000个样本
ntrain = 10000
trainidx = random.sample(range(0,X_train.shape[0]),ntrain)
XT = X_train[trainidx,:,:,:]
print(X_train.shape)#60000，28，28
print(XT.shape)#10000,28,28
###预训练鉴别器########------------------------------------------------------
noise_gan = np.random.uniform(0,1,size=[XT.shape[0],100])#随机生成XT.shape[0]个样本
generated_images = generate.predict(noise_gan)#生成器生成图片
X = np.concatenate(XT,generated_images)#真实图片是XT,生成图片是generate_images
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1#真实图片标签是[1 0]
y[n:,0] = 1#生成图片标签是[0 1]
#鉴别器解冻，使鉴别器可以用
make_trainable(discriminator,True)
#预训练判别器
discriminator.fit(X,y,epochs=1,batch_size=32)
y_hat = discriminator.predict(X)
#计算鉴别器的准确率
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_total = y.shape[0]
n_right = (diff==0).sum()
print( "(%d of %d) right"  % (n_right, n_total))
losses = {"d":[], "g":[]}# 存储生成器和辨别器的训练损失
#  ---------------------- 7、训练 -----------------------------------------
#  ---------------------- 8、输出训练数据 ----------------------------------
def train_for_n(nb_epoch=100,plt_frg=25,BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):
        #生成器生成样本
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gan = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generate.predict(noise_gan)  # generator 生成器
        #鉴别器训练
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1
        make_trainable(discriminator, True)  # 让判别器神经网络各层可用
        d_loss = discriminator.train_on_batch(X, y)  # discriminator 判别器训练
        losses["d"].append(d_loss)  # 存储辨别器损失loss
        #训练生成对抗网络
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1
        # 存储生成器损失loss
        make_trainable(discriminator, False)  # 辨别器的训练关掉
        g_loss = GAN.train_on_batch(noise_tr, y2)  # GAN 生成对抗网络(包括生成器和判别器)训练
        losses["g"].append(g_loss)
        #更新loss图
        if e % plt_frg == plt_frg - 1:
            plot_loss(losses)
            plot_gen()

    train_for_n(nb_epoch=1000, plt_frq=10, BATCH_SIZE=128)
#  ---------------------- 8、输出训练数据 ----------------------------------



