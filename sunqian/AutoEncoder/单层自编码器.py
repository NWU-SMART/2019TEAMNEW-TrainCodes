# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/3
# 文件名称：单层自编码器.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入数据、图像预处理
# 3、构建自编码器模型
# 4、模型训练
# 5、查看自编码器的压缩效果
# 6、查看自编码器的解码效果
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,add
from keras.layers.core import Dense,Activation,Reshape
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入数据、图像预处理 -------------------------------------------
# 数据存在本地E:\\keras_datasets\\mnist.npz
path = 'E:\\keras_datasets\\mnist.npz'
(X_train,_),(X_test,_) = mnist.load_data(path)
# 归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
print('X_train shape：', X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

# np.prod是将28*28矩阵转化成1*784，方便全连接神经网络输入层读取784个神经元
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

#  -------------------------- 2、导入数据、图像预处理--------------------------------------------

#  -------------------------- 3、构建自编码器模型 -------------------------------------------
# 28*28-->8*8-->28*28
input_size = 784
hidden_size = 64
output_size = 784
x = Input(shape=(input_size,))
h = Dense(hidden_size,activation='relu')(x)
r = Dense(output_size,activation='sigmoid')(h)

autoencoder = Model(inputs=x,outputs=r)
autoencoder.compile(optimizer='adam',loss='mse') # 损失函数用均方误差

#  -------------------------- 3、构建自编码器模型 ------------------------------------------

#  -------------------------- 4、模型训练------------------------------------------
epochs = 5
batch_size = 128
history = autoencoder.fit(X_train,X_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=2,
                          validation_data=(X_test,X_test))
#  -------------------------- 4、模型训练------------------------------------------

#  -------------------------- 5、查看自编码器的压缩效果-------------------------------------------
conv_encoder = Model(x,h) # 只取编码器做模型
encoded_imgs = conv_encoder.predict(X_test)

#打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20,8))
for i in range(n):
    # 绘制figure对象的子图(Axes)
    # subplot(numRows,numCols,plotNum) 绘图区域被分成numRows行和numCol列，plotNum执行创建的Axes对象所在的区域
    ax = plt.subplot(1,n,i+1)
    #调用imshow()绘制热图
    plt.imshow(encoded_imgs[i].reshape(4,16).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  -------------------------- 5、查看自编码器的压缩效果 ------------------------------------------

#  -------------------------- 6、查看自编码器的解码效果 ------------------------------------------
decoded_imgs = autoencoder.predict(X_test)
n = 10
plt.figure(figsize=(20,6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3,n,i+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 打印解码图
    ax = plt.subplot(3,n,i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  -------------------------- 6、查看自编码器的解码效果-------------------------------------------

# ----------------------------7、训练过程可视化------------------------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper right')
plt.savefig('SingleLayerAutoEncoder_valid_loss.png')
plt.show()
# ----------------------------7、训练过程可视化-----------------------------------------------