# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月15日
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建正则化自编码器模型
# 4、训练模型
# 5、模型可视化
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
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
f.close()
print(X_train.shape)
print(X_test.shape)
X_train = X_train.astype('float32')/255#数据预处理，归一化
X_test = X_test.astype('float32')/255
print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
# np.prod是将28X28矩阵转化成1X784向量，方便BP神经网络输入层784个神经元读取
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建自编码器模型 ------------------------------
'''正则化自编码器:在原始单层自编码器上加了正则化项，通过比较正则化自编码器的损失图和单层自编码器的损失图可以观察到，
加了正则化项以后损失变化是平缓的,也就是说加了正则化可以使模型更加平滑，解决了模型过拟合的问题'''
# kernel_regularizer：施加在权重上的正则项
# bias_regularizer：施加在偏置向量上的正则项
# activity_regularizer：施加在输出上的正则项
input = Input(shape=(784,))
hidden = Dense(32,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input)#L1正则
output = Dense(784,activation='sigmoid')(hidden)
autoencoder = Model(inputs=input,outputs=output)
autoencoder.compile(optimizer='adam',loss='mse')
#  ---------------------  3、构建自编码器模型 --------------------------------
#  ---------------------- 4、模型训练 ----------------------------------------
epochs=15
batch_size=128
history =autoencoder.fit(X_train,X_train,
                         batch_size=128,
                         epochs=5, verbose=1,
                         validation_data=(X_test, X_test))
#  ---------------------- 4、模型训练 -----------------------------------------
#  -----------------------5、模型可视化----------------------------------------
from keras.utils import plot_model
# 保存模型
autoencoder.save('keras_NormalizeAutoEnconder.h5')
# 模型可视化
plot_model(autoencoder, to_file='keras_NormalizeAutoEnconder.png', show_shapes=True)
#  -----------------------5、模型可视化----------------------------------------
#  ---------------------- 6、查看自编码器的解码效果 ---------------------------
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
#  ---------------------- 6、查看自编码器的解码效果 ---------------------------
#  -----------------------7、训练过程可视化 ----------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()
#  -------------------------7、训练过程可视化 ---------------------------------

