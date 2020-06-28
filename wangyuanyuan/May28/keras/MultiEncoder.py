#-----------------------------------------开发者信息--------------------------------
#开发者：王园园
#开发日期：2020.1.2 MLP-招聘信息文本分类
#开发软件：pycharm
#项目：多层自编码器（keras）

#-----------------------------------------代码布局-----------------------------------
#1、导入包
#2、读取手写体数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看自编码器的压缩效果
#7、查看自编码器的解码效果

#---------------------------------------------导包-------------------------------------
import numpy as np
from keras import Input, Sequential, Model
from keras.backend import shape
from keras.layers import Dense
from networkx.drawing.tests.test_pylab import plt

#------------------------------------------读取手写体数据及与图像预处理--------------------
#60000个训练，10000个测试
path = 'D:/keras_datasets/mnist.npz'
f = np.load(path)
x_train = f['x_train']
x_test = f['x_test']
f.close()

#数据预处理，归一化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#数据准备,将28*28矩阵转换成1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

#------------------------------------------构建多层自编码器模型----------------------
input_size = 784
hidden_size = 128
code_size = 64

#定义神经网络层数
x = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(x)
h = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(h)
r = Dense(input_size, activation='sigmoid')(hidden_2)
autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adam', loss='mse')

#---------------------------------------------模型训练与可视化------------------------------
epochs = 5
batch_size = 128
history = autoencoder.fit(x_train, x_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, x_test))

#------------------------------------------查看自编码器的压缩效果-----------------------------
#为隐藏层的结果（encoder的最后一层），只取编码器做模型
conv_encoder = Model(x, h)
encoded_imgs = autoencoder.predict(x_test)
#打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#------------------------------------------查看自编码器的解码效果------------------------------
decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20,6))
for i in range(n):
    #原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #解码效果图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show

#----------------------------------------------训练过程可视化---------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


