# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.9
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/
# /------------------ 代码布局 --------------------*/
# 1.导入需要的包
# 2.读取图片数据
# 3.图片预处理
# 4.超参数建立
# 4.生成器和判别器模型建立
# 5.训练
# 6.训练结果可视化
# 7.查看效果
# /------------------ 代码布局 --------------------*/


# /------------------ 导入需要的包--------------------*/
import numpy as np
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
# /------------------ 导入需要的包--------------------*/


# /------------------ 读取数据--------------------*/
# 数据路径
path = 'mnist.npz'
data = np.load(path)
# ['x_test', 'x_train', 'y_train', 'y_test']
# print(data.files)
# 读取数据
x_train = data['x_train']#(60000, 28, 28)
x_test = data['x_test']#(10000, 28, 28)
data.close()
# /------------------ 读取数据--------------------*/
# /------------------ 数据预处理 --------------------*/

# 归一化操作
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(60000,1,28,28)
x_test = x_test.reshape(10000,1,28,28)

# /------------------ 数据预处理 --------------------*/
#  --------------------- 超参数设置 ---------------------
# 输入、隐藏和输出层神经元个数 (1个隐藏层)

dropout_rate = 0.25
from keras.optimizers import Adam
# Optim优化器
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)

#  --------------------- 超参数设置 ---------------------
# /------------------ 生成器和判别器模型建立 --------------------*/
# 定义输入图片的形式，通道数是第一个输入的
# import tensorflow as tf
# tf.keras.backend.set_image_data_format('channels_first')
from keras import backend as K
K.set_image_data_format('channels_first')
K.image_data_format()
# 随机噪声通过生成器生成一个假的图片


# 生成器建立
from keras.layers import Dense,\
    BatchNormalization,Activation,\
    Conv2D,Input,Reshape,UpSampling2D
from keras.models import Model

input = Input(shape=[200])
#Glorot正态分布初始化权重,它从以 0 为中心，
# 标准差为 stddev = sqrt(2 / (fan_in + fan_out)) 的截断正态分布中抽取样本，
# 其中 fan_in 是权值张量中的输入单位的数量， fan_out 是权值张量中的输出单位的数量。
# 200->(39200)200*14*14
Hidden = Dense(units=200*14*14,kernel_initializer='glorot_normal')(input)
Hidden = BatchNormalization()(Hidden)
Hidden = Activation('relu')(Hidden)

# 将转换为的Hidden的矩阵，转换为图片类型
Hidden = Reshape([200,14,14])(Hidden)
# 上采样将图片转换为200*28*28
Hidden = UpSampling2D(size=(2,2))(Hidden)
# 将通道数转为1 200->100->50->1
# 转换为100
result_100 = Conv2D(filters=100,kernel_size=(3,3),
                    padding='same',kernel_initializer='glorot_normal')(Hidden)
result_100 = BatchNormalization()(result_100)
result_100 = Activation('relu')(result_100)
# 转换为50
result_50 = Conv2D(filters=50,kernel_size=(3,3),
                    padding='same',kernel_initializer='glorot_normal')(result_100)
result_50 = BatchNormalization()(result_50)
result_50 = Activation('relu')(result_50)
# 转换为1
result = Conv2D(filters=1,kernel_size=(1,1),
                    padding='same',kernel_initializer='glorot_normal')(result_50)
result = Activation('sigmoid')(result)
# 模块定义
generator = Model(input,result)
generator.compile(loss='binary_crossentropy',optimizer=opt)
# 输出生成器框架
generator.summary()


# 判别器建立(提取特征)
# 通过将特征转换维度最终输出一个两位的数据判断是否是真或假
from keras.layers import LeakyReLU,Dropout,Flatten,MaxPool2D
d_input = Input(shape=(1,28,28))
# 1*28*28->256*28*28
H = Conv2D(filters=256,kernel_size=(3,3),
           activation='relu',strides=(1,1),
           padding='same')(d_input)
# 256*28*28->256*14*14
H = MaxPool2D(pool_size=2,padding='same')(H)
# relu函数的一种变体
H = LeakyReLU(0.2)(H)
H = Dropout(0.25)(H)
# 256*14*14->512*14*14
H = Conv2D(filters=512,kernel_size=(3,3),
           activation='relu',strides=(1,1),
           padding='same')(H)
# 512*14*14->512*7*7
H = MaxPool2D(pool_size=2,padding='same')(H)
# relu函数的一种变体
H = LeakyReLU(0.2)(H)
H = Dropout(0.25)(H)
# 转换为一维数据25088
H = Flatten()(H)

#将25088->256->2
# 25088->256
H = Dense(units=256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(0.25)(H)
# 256->2
result_d = Dense(units=2,activation='softmax')(H)
# 定义判别器模块
discrimination = Model(d_input,result_d)
discrimination.compile(loss='categorical_crossentropy',optimizer=dopt)
discrimination.summary()

# --------------------- 构造GAN ---------------------

#
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discrimination,False)
# 构建GAN网络
input_gan = Input(shape=[200])
# 通过生成器生成新图像1*28*28
new_gen = generator(input_gan)
# 通过判别器判断生成器生成的图像1*28*28->2
result_last = discrimination(new_gen)
# 根据输入输出构建模型
GAN = Model(input_gan,result_last)
GAN.compile(loss='categorical_crossentropy',optimizer=opt)
GAN.summary()

# --------------------- 构造GAN ---------------------

# ---------------------结果显示---------------------
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
# 定义显示损失函数
def plt_loss(loss_):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(loss_["d"], label='discriminitive loss')
    plt.plot(loss_["g"], label='generative loss')
    plt.legend()
    plt.show()
#  描绘生成器生成图像
def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    # 随机生成0-1的随机数
    noise = np.random.uniform(0, 1, size=[n_ex, 200])
    # 将噪声输入到生成器中
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    # 将结果显示
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('genrated_images.jpg')
    plt.tight_layout()
    plt.show()


# 预训练辨别器,生成1万个随机样本
noise_gen = np.random.uniform(0,1,size=[10000,200]) # 生成XT.shape[0]个随机样本
generated_images = generator.predict(noise_gen)  # 生成器产生图片样本
# 取出10000个真实样本
ntrain = 10000
import random
trainidx = random.sample(range(0,x_train.shape[0]), ntrain)
XT = x_train[trainidx,:,:,:]
# 将真实图像和判别图像通过判别器判断
x = np.concatenate((XT, generated_images))
n = XT.shape[0]
# 构建标签
y = np.zeros([2*n,2])  # 构造辨别器标签 one-hot encode
y[:n,1] = 1  # 真实图像标签 [1 0]
y[n:,0] = 1  # 生成图像标签 [0 1]
make_trainable(discrimination,True)
predict = discrimination.fit(x,y,epochs=1,batch_size=32)

y_hat = discrimination.predict(x)

#  计算辨别器的准确率
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_total = y.shape[0]
n_right = (diff==0).sum()

print( "(%d of %d) right"  % (n_right, n_total))
# 存储生成器和辨别器的训练损失
loss_ = {"d":[], "g":[]}
# 开始训练
def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        # 随机生成噪声
        image_batch = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 200])
        generated_images = generator.predict(noise_gen)  # generator 生成器

        # 训练判别器
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discrimination, True)  # 让判别器神经网络各层可用
        d_loss = discrimination.train_on_batch(X, y)  # discriminator 判别器训练
        loss_["d"].append(d_loss)  # 存储辨别器损失loss

        # 调用GAN网络训练
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 200])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        # 存储生成器损失loss
        make_trainable(discrimination, False)  # 辨别器的训练关掉
        g_loss = GAN.train_on_batch(noise_tr, y2)  # GAN 生成对抗网络(包括生成器和判别器)训练
        loss_["g"].append(g_loss)
        ### ------ 训练生成对抗网络 ------------


        # 更新损失loss图
        if e % plt_frq == plt_frq - 1:
            plt_loss(loss_)
            plot_gen()
# 调用韩散户训练
train_for_n(nb_epoch=1000, plt_frq=10,BATCH_SIZE=128)