# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/10
#  @Author: MiJizong
#  @Content: 多层自编码器——Keras三种方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器三种模型
# 4、训练
# 5、模型可视化
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers, Sequential
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)

path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train = f['x_train']
# 测试数据
X_test = f['x_test']
f.close()
# 数据放到本地路径

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)  # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# 输入、隐藏和输出层神经元个数 (3个隐藏层)
input_size = 784
hidden_size = 128
code_size = 64  # dimension 784 = (28*28) --> 128 --> 64 --> 128 --> 784 = (28*28)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3.1、构建多层自编码器Sequential模型 ----------------
autoencoder1 = Sequential()
autoencoder1.add(Dense(hidden_size,activation='relu'))    # 全连接 784 → 128
autoencoder1.add(Dense(code_size,activation='relu'))      # 128 → 64
autoencoder1.add(Dense(hidden_size,activation='relu'))    # 64 → 128
autoencoder1.add(Dense(input_size,activation='sigmoid'))  # 128 → 784
autoencoder1.compile(optimizer='adam',loss='mse')

#  --------------------- 3.1、构建多层自编码器Sequential模型 ----------------

#  --------------------- 3.2、构建多层自编码器API模型 -----------------------
# 定义神经网络层数
x = Input(shape=(input_size,))   # 784

hidden_1 = Dense(hidden_size, activation='relu')(x)     # 全连接 784 → 128
h = Dense(code_size, activation='relu')(hidden_1)       # 128 → 64
hidden_2 = Dense(hidden_size, activation='relu')(h)     # 64 → 128
r = Dense(input_size, activation='sigmoid')(hidden_2)   # 128 → 784

# 构建模型，给定模型优化参数
autoencoder2 = Model(inputs=x, outputs=r)
autoencoder2.compile(optimizer='adam', loss='mse')

#  --------------------- 3.2、构建多层自编码器API模型 -----------------------

#  --------------------- 3.3、构建多层自编码器class继承模型 ----------------
# 定义神经网络层数                         # 若使用class继承模型 则145行可视化SVG()需要注释掉
                                          #  否则回报错‘AttributeError: 'Coder' object has no attribute '_network_nodes'’
inputs = Input(shape=(input_size,))      # 784 * 1
class Coder(keras.Model):
    def __init__(self):
        super(Coder, self).__init__()
        self.dense1 = Dense(hidden_size,activation='relu')     # 全连接 784 → 128
        self.dense2 = Dense(code_size,activation='sigmoid')    # 128 → 64
        self.dense3 = Dense(hidden_size, activation='relu')    # 64 → 128
        self.dense4 = Dense(input_size, activation='sigmoid')  # 128 → 784

    def call(self, inputs):     # 注意：此处输入变量的名字最好不要用x，会报错，会与151行中的‘onv_encoder = Model(x, h) ’的x变量名冲突
        t = self.dense1(inputs)
        t = self.dense2(t)
        t = self.dense3(t)
        t = self.dense4(t)
        return t

autoencoder3 = Coder()  # 实例化
print(autoencoder3)

autoencoder3.compile(optimizer='adam',loss='mse')  # 模型编译
#  --------------------- 3.3、构建多层自编码器class继承模型 ----------------

#  --------------------- 4、训练 ------------------------------------------

# 设定epochs和batch_size大小
epochs = 5
batch_size = 128

# 训练模型
history = autoencoder1.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,   # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                          validation_data=(X_test, X_test)
                          )

#  --------------------- 4、训练 ------------------------------------------

#  --------------------- 5、模型可视化 -------------------------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(autoencoder1).create(prog='dot', format='svg'))

#  --------------------- 5、模型可视化 -------------------------------------


#  --------------------- 6、查看自编码器的压缩效果 -------------------------

# 为隐藏层的结果 (encoder的最后一层)
conv_encoder = Model(x, h)  # 只取编码器做模型  (取输入层x和隐藏层h，作为网络结构)
encoded_imgs = conv_encoder.predict(X_test)  # 获取压缩之后的信号

# 打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))  # figsize用来设置图形的大小，20为图形的宽， 8为图形的高，单位为英寸

for i in range(n):
    ax = plt.subplot(1, n, i + 1)                 # 规定在面板的哪个区域呈现图像
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)  # 8*8 的特征，转化为 4*16的图像,查看图片的压缩效果
    plt.gray()                                    # 灰度展示
    ax.get_xaxis().set_visible(False)             # 不显示x轴
    ax.get_yaxis().set_visible(False)             # 不显示x轴
plt.show()

#  --------------------- 6、查看自编码器的压缩效果 --------------------------

#  --------------------- 7、查看自编码器的解码效果 --------------------------

# decoded_imgs 为输出层的结果
decoded_imgs = autoencoder1.predict(X_test)

n = 10
plt.figure(figsize=(20, 6))  # figsize用来设置图形的大小，20为图形的宽， 6为图形的高，单位为英寸
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 解码效果图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

#  --------------------- 7、查看自编码器的解码效果 --------------------------

#  --------------------- 8、训练过程可视化 ---------------------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 8、训练过程可视化 ----------------------------------
