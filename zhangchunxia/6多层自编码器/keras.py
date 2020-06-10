# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月10日
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、训练模型
# 5、查看自编码器的压缩效果
# 6、查看自编码器的解码效果
# 7、训练过程可视化
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
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
#  ---------------------- 1、导入需要包 ------------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
path = '...'
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
print(X_train.shape)
print(X_test.shape)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
# np.prod是将28X28矩阵转化成1X784向量，方便BP神经网络输入层784个神经元读取
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建多层自编码器模型 ---------------------------
input_size= 784
hidden_size = 128
code_size = 64 #dimension 784 = (28*28) --> 128 --> 64 --> 128 --> 784 = (28*28)
#/-------------------------------API方式--------------------------------------
x = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size,activation='relu')(x)
hidden_2 = Dense(code_size,activation='relu')(hidden_1)
hidden_3 = Dense(hidden_size,activation='relu')(hidden_2)
output = Dense(input_size,activation='sigmoid')(hidden_3)
autoencoder = Model(inputs=x,outputs=output)
autoencoder.compile(optimizer='adam',loss='mse')
#/-------------------------------API方式--------------------------------------
#/-------------------------------class方式------------------------------------
class autoencoder(keras.Model):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.hidden1 = keras.layer.Dense(hidden_size,activation='relu')
        self.hidden2= keras.layer.Dense(code_size, activation='relu')
        self.hidden3 = keras.layer.Dense(hidden_size, activation='relu')
        self.output = keras.layer.Dense(input_size, activation='sigmoid')
    def call(self,inputs):
      x = self.hidden1(inputs)
      x = self.hidden2(x)
      x = self.hidden3(x)
      x = self.output(x)
      return x
autoencoder.compile(optimizer='adam',loss='mse')
#/-------------------------------class方式------------------------------------
#/-------------------------------序贯方式-------------------------------------
from keras.models import Sequential
model = Sequential()
model.add(Dense(hidden_size,activation='relu'))
model.add(Dense(code_size,activation='relu'))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dense(input_size,activation='sigmoid'))
model.compile(optimizer='adam',loss='mse')
#/-------------------------------序贯方式-------------------------------------
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
#  ----------------------------- 4、模型训练 ----------------------------------
history = autoencoder.fit(X_train,X_train,batch_size=128,epochs=5,verbose=1,validation_data=(X_test,X_test))
#  ----------------------------- 4、模型训练 ----------------------------------
#  ----------------------------- 5、查看自编码器的压缩效果 ---------------------
conv_encoder = Model(x,hidden_2)
encoded_imgs = conv_encoder.predict(X_test)
n = 10#用10张测试集的手写体查看压缩效果
plt.figure(figsize=(20,8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4,16).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  ----------------------------- 5、查看自编码器的压缩效果 ---------------------
#  ----------------------------- 6、查看自编码器的解码效果 ---------------------
decoded_imgs = autoencoder.predict(X_test)
n = 10
plt.figure(figsize=(20,6))
for i in range(n):
    #原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
   #解码后的图
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  -----------------------------6、查看自编码器的解码效果 ---------------------
#  ---------------------------- 7、训练过程可视化 -----------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train','validation',loc='upper left')
plt.show()
#  ---------------------------- 7、训练过程可视化 -----------------------------