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
# 3、构建自编码器模型
# 4、训练模型
# 5、查看自编码器的解码效果
# 6、训练过程可视化
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
#数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')/255#数据预处理，归一化
X_test = X_test.astype('float32')/255
print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#加噪,生成噪声，利用正态分布，0.5*均值为0方差为1的正太分布，加入到原始的数据中,生成新的数据
'''
loc：float  概率分布的均值（对应着整个分布的中心centre）
scale：float 概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints   输出的shape，默认为None，只输出一个值
'''
noise_factor=0.5
X_train_noisy =X_train + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=X_train.shape)
X_test_noisy =X_test + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=X_test.shape)
X_train_noisy=np.clip(X_train_noisy,0.,1.)
X_test_noisy=np.clip(X_test_noisy,0.,1.)
#np.clip的相关参数
'''
第一个参数表示数组
第二个参数表示代替数组中最小的数，比0小替换为0
第三个参数表示代替数组中最大的数，比1大替换为1
'''
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建卷积自编码器模型 ---------------------------
x=Input(shape=(28,28,1))
#编码器
conv1_1 = Conv2D(32,(3,3),activation='relu',padding='same')(x)# 28*28*1 --> 28*28*32
pool1 = MaxPooling2D(pool_size = (2,2),padding='same')(conv1_1)# 28*28*32 --> 14*14*32
conv1_2 = Conv2D(32,(3,2),activation='relu',padding='same')(pool1)# 14*14*32 --> 14*14*32
h = MaxPooling2D(pool_size = (2,2),padding='same')(conv1_2) # 14*14*32 --> 7*7*32
#解码器
conv2_1 =Conv2D(32,(3,3),activation='relu',padding='same')(h) # 7*7*32 --> 7*7*32
up1 = UpSampling2D((2,2))(conv2_1)# 7*7*32 --> 14*14*32
conv2_2 = Conv2D(32,(3,3),activation='relu',padding='same')(up1) # 14*14*32 --> 14*14*32
up2 = UpSampling2D((2,2))(conv2_2)# 14*14*32 --> 28*28*32
r = Conv2D(1,(3,3),activation='sigmoid',padding='same')(up2)# 28*28*32 --> 28*28*1
autoencoder = Model(inputs=x,outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#  ---------------------  3、构建卷积自编码器模型 ---------------------------
#  ---------------------  4、模型训练----------- ---------------------------
epochs = 3
batch_size = 128
history = autoencoder.fit(X_train, X_train, batch_size=batch_size,
                      epochs = epochs,
                      verbose = 1,
                      validation_data=(X_test, X_test))
from keras.utils import plot_model
# 保存模型
autoencoder.save('keras_DenoiseAutoEnconder.h5')
# 模型可视化
plot_model(autoencoder, to_file='keras_DenoiseAutoEnconder.png', show_shapes=True)
#  ---------------------  4、模型训练----------- ---------------------------
#  -----------------------5、查看解码效果 ----------------------------------
decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
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
#  -----------------------5、查看解码效果 ----------------------------------
#  -----------------------6、训练过程可视化 --------------------------------
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#  -----------------------6、训练过程可视化 --------------------------------
