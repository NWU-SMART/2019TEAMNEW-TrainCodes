# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月8日
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型训练
# 5、模型可视化
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -------------------------------
from keras import Sequential
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
#  ---------------------- 1、导入需要包 -------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 ---------------------
path = '...'#文件路径，以npz结尾的文件里面还有四个文件，x_test,y_test,x_train,y_train
f = np.load(path)#用f.file()读取里面的文件
X_train = f['x_train']#60000个训练
X_test =  f['x_test']#10000个测试
f.close()
#查看训练集和测试集的维度
#输出的结果
#(60000, 28, 28)
#(10000, 28, 28)
print(X_train.shape)
print(X_test.shape)
#数据预处理/归一化
X_train = X_train.astype("float32")/255#将像素点转化到0——1之间
X_test = X_test.astype("float32")/255#将像素点转化到0——1之间
print('X_train.shape',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#输出结果
#X_train shape: (60000, 28, 28)
#60000 train samples
#10000 test samples
##数据准备
# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))#重塑为600000×28的二维tensor
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))#重塑为100000×28的二维tensor
#  ---------------------  2、读取手写体数据及与图像预处理 ---------------------
#  ---------------------  3、构建单层自编码器模型 -----------------------------
# 输入、隐藏和输出层神经元个数 (1个隐藏层)
input_size = 784
hidden_size = 64
output_size = 784# dimenskion 784 = (28*28) --> 64 --> 784 = (28*28)
#/-------------------------------API方式--------------------------------------
#定义神经网路层数
x = Input(shape=(input_size,))
h = Dense(hidden_size,activation='relu')
r = Dense(output_size,activation='softmax')
autoencoder = Model(inputs=x,outputs=r)
autoencoder.compile(optimizer='adam',loss='mse')#模型优化参数
#/-------------------------------API方式--------------------------------------
#/------------------------------Sequential方式--------------------------------
autoencoder1 = Sequential()
autoencoder1.add(Dense(hidden_size,activation='relu'))
autoencoder1.add(Dense(output_size,activation='softmax'))
autoencoder.compile(optimizer='adam',loss='mse')
#/------------------------------Sequential方式--------------------------------
#/------------------------------class方式-------------------------------------
class autoencoder2(keras.Model):
    def __init__(self):
      super(autoencoder2,self).__init__()
      self.dense1 = keras.layer.Dense(hidden_size,activation='relu'),
      self.dense2 = keras.layer.Dense(output_size,activation='softmax')
    def call(self,inputs):
       x = self.dense1(inputs)
       x = self.dense2(x)
       return x
autoencoder.compile(optimizer='adam',loss='mse')
#/------------------------------class方式-------------------------------------
#  ----------------------------- 4、模型训练 ----------------------------------
#使用x_train作为输入和输出来训练utoencoder，并使用x_test进行validation
history = autoencoder.fit(X_train,X_train,
                          batch_size=128,
                          epochs=5,verbose=1,
                          validation_data=(X_test, X_test))
#  ----------------------------- 4、模型训练 ----------------------------------
#  ----------------------------- 5、模型可视化----------------------------------
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 可视化模型
SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
#  ----------------------------- 5、模型可视化 ----------------------------------
#  ----------------------------- 6、查看自编码器的压缩效果 -----------------------
#只取编码器为模型，取x和h为网络结构
conv_encoder = Model(x,h)
encoded_imgs = conv_encoder.predict(X_test)
#打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20,8))#指定图像的宽和高，单位为英寸
for i in range(n):
    ax = plt.sunplot(1,n,i+1)#表示一次性在figure上创建成1*n的网格
    plt.imshow(encoded_imgs[i].reshape(4,16).T) # 8*8 的特征，转化为 4*16的图像
    plt.gray()#只有黑白两种颜色
    ax.get.xaxis().set.visible(False)#不显示x轴
    ax.get.yaxis().set.visible(False)#不显示y轴
plt.show()
#  ----------------------------- 6、查看自编码器的压缩效果 -----------------------
#  ----------------------------- 7、查看自编码器的解码效果 -----------------------
#使用autoencoder对x_test预测，并将预测结果绘制出来，和原始像进行对比
decoded_imgs = autoencoder.predict(X_test)
n=10
plt.figure(figsize=(20,6))
for i in range(n):#打印原图
    ax = plt.subplot(3,n,i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get.xaxis().set.visible(False)#不显示x轴
    ax.get.yaxis().set.visible(False)#不显示y轴
for i in range(n):#查看解码图
    ax = plt.subplot(3,n,i+1+n)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get.xaxis().set.visible(False)#不显示x轴
    ax.get.yaxis().set.visible(False)#不显示y轴
plt.show()
#  ---------------------------- 8、训练过程可视化 ------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train,validation',loc='upper right')
plt.show()
#  ---------------------------- 8、训练过程可视化 ------------------------------

