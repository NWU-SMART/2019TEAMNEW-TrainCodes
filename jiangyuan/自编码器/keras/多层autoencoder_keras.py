# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年6月9日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#


#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
#  -------------------------- 1、导入需要包 -------------------------------


#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
path = 'mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
# --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建多层自编码器模型 ---------------------
# 输入、隐藏和输出层神经元个数 (1个隐藏层)
input_size = 784
hidden_size = 128
code_size = 64
output_size = 784  # dimenskion 784 = (28*28) --> 64 --> 784 = (28*28)

# 定义神经网络层数
x = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(x)
h = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(h)
output = Dense(output_size, activation='sigmoid')(hidden_2)

# 构建模型，给定模型优化参数
autoencoder = Model(inputs=x, outputs=output)
autoencoder.compile(optimizer='adam', loss='mse')
#  --------------------- 3、构建多层自编码器模型 ---------------------


#  ---------------------4、模型可视化 ---------------------
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# 可视化模型
SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
#  ---------------------4、模型可视化 ---------------------


#  --------------------- 5、模型训练 ---------------------
# 设定peochs和batch_size大小
epochs = 5
batch_size = 128
# 训练模型
history = autoencoder.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test, X_test)
                          )
#  --------------------- 5、模型训练 ---------------------


#  --------------------- 6、查看自编码器的压缩效果 ---------------------
# 为隐藏层的结果
conv_encoder = Model(x, h)  # 只取编码器做模型 (取输入层x和隐藏层h，作为网络结构)
encoded_imgs = conv_encoder.predict(X_test)

# 打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)  # 8*8 的特征，转化为 4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  --------------------- 6、查看自编码器的压缩效果 ---------------------


#  --------------------- 7、查看自编码器的解码效果 ---------------------
# decoded_imgs 为输出层的结果
decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 打印解码图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28)) # 784 转换为 28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  --------------------- 7、查看自编码器的解码效果 ---------------------


#  --------------------- 8、训练过程可视化 ---------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#  --------------------- 8、训练过程可视化 ---------------------


