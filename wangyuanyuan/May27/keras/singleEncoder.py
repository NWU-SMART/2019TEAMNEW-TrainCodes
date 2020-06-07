#-----------------------------------------开发者信息---------------------------------
#开发人：王园园
#开发日期：2020.5.27
#开发软件：pycharm
#项目：单层自编码器（keras）

#--------------------------------------------代码布局------------------------------
#1、导入包
#2、读取手写题数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看自编码器的压缩效果
#7、查看自编码器的解码效果

#-------------------------------------------导包------------------------------------
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.utils import model_to_dot
from networkx.drawing.tests.test_pylab import plt

#-------------------------------------------读取数据及数据预处理------------------------
path='D:/keras_datasets/mnist.npz'
f = np.load(path)
x_train = f['x_train']
x_test = f['x_']
f.close()

#观察x_train和x_test维度
print(x_train.shape)       #(60000, 28, 28)
print(x_test.shape)        #(100000, 28, 28)

#数据标准化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#数据准备
#np.prod是将28*28矩阵转换成1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

#----------------------------------------------构建单层自编码器模型---------------------------
#输入层、隐藏层、输出层神经元个数
input_size = 784
hidden_size = 64
output_size = 784

#定义神经网络层数,构建模型优化参数
model = Sequential()
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(output_size, activation='sigmoid'))
autoencoder = model.compile(optimizer='adam', loss='mse')

#-------------------------------------------------训练模型-------------------------------------
epochs = 5
batch_size = 128
history = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test, x_test))

#----------------------------------------------------模型可视化---------------------------------
SVG(model_to_dot(autoencoder).create(prog='dot',format='svg'))

#----------------------------------------------------查看自编码器的压缩效果------------------------
#只取编码器做模型（取输入层x和隐藏层h，作为网络结构）
encoded_imgs = model.predict(x_test)
#打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T) #将8*8的特征转化为4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#----------------------------------------------------查看自编码器的解码效果-------------------------
decoded_imgs = autoencoder.predict(x_test)    #decoded_imgs为输出层的效果
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    #打印原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #打印解码图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))  #784转换为28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#----------------------------------------------------训练过程可视化---------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylable('loss')
plt.xlable('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

